import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, flash, abort, jsonify
)
from flask_bootstrap import Bootstrap
from flask_login import (
    LoginManager, UserMixin, login_user,
    logout_user, login_required, current_user
)
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from PIL import Image
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from torchvision import transforms
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from wtforms import (
    FileField, SubmitField, FloatField, HiddenField,
    StringField, PasswordField
)
from wtforms.validators import DataRequired, Length

from utils.utils import adaptive_instance_normalizaiton
from utils.models import VGGEncoder, Decoder

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Configuration
# --------------------------------------------------
load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
if not app.config["SECRET_KEY"]:
    raise RuntimeError("Set SECRET_KEY in your .env file.")

csrf = CSRFProtect(app)


def normalise_database_url(url):
    """Make hosted-Postgres URLs work with the psycopg (v3) driver.

    Handles: postgres://, postgresql://  (-> postgresql+psycopg://)
    Leaves sqlite:// and explicit driver URLs (postgresql+psycopg2://) alone.
    """
    url = url.strip()
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


database_url = os.environ.get("DATABASE_URL")
if not database_url:
    # Local-development fallback so the app still starts without Postgres.
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{Path(app.instance_path) / 'app.db'}"
    logger.warning("DATABASE_URL not set; using local SQLite database.")

database_url = normalise_database_url(database_url)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# FIX: hosted Postgres (Render, Neon, Supabase, Heroku...) silently closes idle
# connections -> "server closed the connection unexpectedly" on the next request.
# pool_pre_ping tests a connection before use; pool_recycle retires old ones.
if not database_url.startswith("sqlite"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 280,
        "pool_size": 5,
        "max_overflow": 5,
    }

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
# The page is now interactive (AJAX) and can stay open for a long time; a CSRF
# token with the default 1-hour limit would make the next transfer fail.
app.config["WTF_CSRF_TIME_LIMIT"] = None
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Keep uploaded files outside the public static directory.
Path(app.instance_path).mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT = Path(app.instance_path) / "user_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

db = SQLAlchemy()
db.init_app(app)
migrate = Migrate(app, db)  # schema migrations: flask db init / migrate / upgrade
Bootstrap(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Please log in to use style transfer."
login_manager.init_app(app)


# --------------------------------------------------
# Database models
# --------------------------------------------------
class User(UserMixin, db.Model):
    # FIX: "user" is a reserved word in PostgreSQL. It only worked because
    # SQLAlchemy quotes it; any raw SQL / psql / BI tool breaks. Use "users".
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    transfers = db.relationship(
        "StyleTransfer",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class StyleTransfer(db.Model):
    __tablename__ = "style_transfers"

    id = db.Column(db.Integer, primary_key=True)
    # FIX: ondelete="CASCADE" so the DB itself removes history rows when a user
    # is deleted (the ORM cascade alone doesn't cover raw/bulk deletes).
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    content_filename = db.Column(db.String(255), nullable=False)
    style_filename = db.Column(db.String(255), nullable=False)
    output_filename = db.Column(db.String(255), nullable=False)
    alpha = db.Column(db.Float, nullable=False, default=1.0)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,  # history is ordered by this column
    )


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None
    except SQLAlchemyError:
        db.session.rollback()
        logger.exception("Database error while loading user")
        return None


# FIX: after any failed DB operation the session is left in a broken state;
# always return the connection to the pool cleanly at the end of a request.
@app.teardown_request
def cleanup_session(exc):
    if exc is not None:
        db.session.rollback()


def is_ajax():
    """True for fetch/XHR calls made by the interactive front-end."""
    return (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.accept_mimetypes.best == "application/json"
    )


@app.errorhandler(SQLAlchemyError)
def handle_db_error(exc):
    db.session.rollback()
    logger.exception("Unhandled database error")
    message = "A database error occurred. Please try again shortly."
    if is_ajax():
        return jsonify(ok=False, error=message), 500
    return render_template_string_safe(message), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(exc):
    message = "Upload too large: the two images together must be under 10 MB."
    if is_ajax():
        return jsonify(ok=False, error=message), 413
    flash(message, "danger")
    return redirect(url_for("index"))


def render_template_string_safe(message):
    from markupsafe import escape
    return f"<h3>{escape(message)}</h3>"


# --------------------------------------------------
# Forms
# --------------------------------------------------
class RegistrationForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=80)])
    email = StringField("Email", validators=[DataRequired(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=128)])
    submit = SubmitField("Create account")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Log in")


class UploadForm(FlaskForm):
    content = FileField("Content_Image")
    style = FileField("Style_Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")


# --------------------------------------------------
# Authentication routes
# --------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        email = form.email.data.strip().lower()

        if "@" not in email or email.startswith("@") or email.endswith("@"):
            flash("Enter a valid email address.", "danger")
            return render_template("register.html", form=form)

        existing = db.session.execute(
            db.select(User).where(
                db.or_(
                    db.func.lower(User.username) == username.lower(),
                    db.func.lower(User.email) == email
                )
            )
        ).scalars().first()  # FIX: scalar_one_or_none() raises if 2 rows match
                             # (one row matches username, another matches email)
        if existing:
            flash("That username or email is already registered.", "danger")
            return render_template("register.html", form=form)

        user = User(username=username, email=email)
        user.set_password(form.password.data)
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            # FIX: two simultaneous sign-ups can both pass the check above;
            # the unique constraint then fires. Handle it instead of a 500.
            db.session.rollback()
            flash("That username or email is already registered.", "danger")
            return render_template("register.html", form=form)
        except SQLAlchemyError:
            db.session.rollback()
            logger.exception("Registration failed")
            flash("Could not create the account right now. Try again.", "danger")
            return render_template("register.html", form=form)

        login_user(user)
        flash("Account created successfully.", "success")
        return redirect(url_for("index"))

    return render_template("register.html", form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = db.session.execute(
            db.select(User).where(db.func.lower(User.email) == email)
        ).scalars().first()

        if user and user.check_password(form.password.data):
            login_user(user)
            flash("Welcome back!", "success")
            return redirect(url_for("index"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html", form=form)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have logged out.", "success")
    return redirect(url_for("login"))


# --------------------------------------------------
# AdaIN model initialisation
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIX: paths were relative to the current working directory, so running
# gunicorn from anywhere but the repo root crashed at import time.
encoder = VGGEncoder(str(BASE_DIR / "vgg_normalised.pth")).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(
    torch.load(
        BASE_DIR / "experiment" / "experiment1" / "decoder_90.pth",
        map_location=device
    )
)
encoder.eval()
decoder.eval()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def user_upload_dir():
    directory = UPLOAD_ROOT / str(current_user.id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_uploaded_file(file_storage):
    """Save a new image with a unique name in the user's own directory."""
    original_name = secure_filename(file_storage.filename or "")
    if not original_name or not allowed_file(original_name):
        raise ValueError("Upload a PNG, JPG, or JPEG image.")
    extension = original_name.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"
    file_storage.save(user_upload_dir() / filename)
    return filename


def existing_user_image(filename):
    """Only allow previously saved images belonging to this user."""
    if not filename or Path(filename).name != filename:
        return None
    path = user_upload_dir() / filename
    return path if path.is_file() else None


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    style_transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feat = encoder(content_image, is_test=True)
        style_feat = encoder(style_image, is_test=True)
        stylized_feat = adaptive_instance_normalizaiton(content_feat, style_feat)
        stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat
        stylized_image = decoder(stylized_feat)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone().squeeze(0).clamp(0, 1)
    transforms.ToPILImage()(image).save(path)


# --------------------------------------------------
# Main style-transfer page
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None
    history_id = None
    alpha = None

    if request.method == "POST" and not form.validate_on_submit():
        error = "Your session expired or the form was invalid. Reload the page and try again."
        if is_ajax():
            return jsonify(ok=False, error=error, fields=form.errors), 400

    if form.validate_on_submit():
        result_path = None
        try:
            if form.content.data and form.content.data.filename:
                content_filename = save_uploaded_file(form.content.data)
            else:
                content_filename = form.content_path.data

            if form.style.data and form.style.data.filename:
                style_filename = save_uploaded_file(form.style.data)
            else:
                style_filename = form.style_path.data

            content_path = existing_user_image(content_filename)
            style_path = existing_user_image(style_filename)

            if content_path is None:
                raise ValueError("Please upload a valid content image.")
            if style_path is None:
                raise ValueError("Please upload a valid style image.")

            alpha = float(form.alpha.data)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("Style strength must be between 0 and 1.")

            with Image.open(content_path) as image:
                content_image = image.convert("RGB")
            with Image.open(style_path) as image:
                style_image = image.convert("RGB")

            stylized_image = style_transfer(
                content_image, style_image, encoder, decoder, alpha, device
            )

            result_filename = f"{uuid.uuid4().hex}_result.png"
            result_path = user_upload_dir() / result_filename
            save_image(stylized_image, result_path)

            # Record history only after successful image generation.
            record = StyleTransfer(
                user_id=current_user.id,
                content_filename=content_filename,
                style_filename=style_filename,
                output_filename=result_filename,
                alpha=alpha
            )
            db.session.add(record)
            db.session.commit()
            history_id = record.id
            result_image = result_filename

        except ValueError as exc:
            # User-facing validation errors are safe to show.
            db.session.rollback()
            error = str(exc)

        except SQLAlchemyError:
            # FIX: previously str(exc) of a DB error (with SQL, table names,
            # sometimes the connection string) was shown to the user, and the
            # generated result file was left on disk with no DB record.
            db.session.rollback()
            logger.exception("Could not save style-transfer history")
            if result_path is not None and result_path.exists():
                result_path.unlink(missing_ok=True)
            error = "Could not save your result to the database. Please try again."

        except Exception:
            db.session.rollback()
            logger.exception("Style transfer failed")
            error = "Style transfer failed. Please try different images."

    # FIX: the hidden content_path/style_path fields were never filled in, so
    # "reuse the images I already uploaded" always failed with "Please upload a
    # valid content image." Re-populate them so a second run works.
    if content_filename:
        form.content_path.data = content_filename
    if style_filename:
        form.style_path.data = style_filename

    if is_ajax() and request.method == "POST":
        def img_url(name):
            return url_for("send_image", filename=name) if name else None

        return jsonify(
            ok=bool(result_image),
            error=error,
            history_id=history_id,
            alpha=alpha,
            content_filename=content_filename,
            style_filename=style_filename,
            result_filename=result_image,
            content_url=img_url(content_filename),
            style_url=img_url(style_filename),
            result_url=img_url(result_image),
        ), (200 if result_image else 400)

    initial = {
        "result_url": url_for("send_image", filename=result_image) if result_image else None,
        "content_url": url_for("send_image", filename=content_filename) if result_image and content_filename else None,
        "alpha": alpha if result_image else None,
        "error": error,
    }

    return render_template(
        "index.html",
        form=form,
        initial=initial,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# --------------------------------------------------
# User-specific image access and history
# --------------------------------------------------
@app.route("/uploads/<filename>")
@login_required
def send_image(filename):
    path = existing_user_image(filename)
    if path is None:
        abort(404)
    return send_from_directory(user_upload_dir(), filename)


@app.route("/history")
@login_required
def history():
    page = request.args.get("page", 1, type=int)
    # FIX: the query was unbounded (loads every row for the user). Paginate.
    pagination = db.paginate(
        db.select(StyleTransfer)
        .where(StyleTransfer.user_id == current_user.id)
        .order_by(StyleTransfer.created_at.desc()),
        page=page, per_page=20, error_out=False,
    )
    transfers = pagination.items

    # FIX: DB rows can outlive their files (ephemeral disks on Render/Heroku,
    # manual cleanup). Flag them so the template can avoid broken <img> tags:
    #   {% if t.files_ok %} <img ...> {% else %} file no longer available {% endif %}
    for t in transfers:
        t.files_ok = all(
            existing_user_image(name) is not None
            for name in (t.content_filename, t.style_filename, t.output_filename)
        )

    return render_template(
        "history.html", transfers=transfers, pagination=pagination
    )


@app.route("/history/<int:transfer_id>/delete", methods=["POST"])
@login_required
def delete_transfer(transfer_id):
    # Ownership is enforced in the query: you can only delete your own rows.
    record = db.session.execute(
        db.select(StyleTransfer).where(
            StyleTransfer.id == transfer_id,
            StyleTransfer.user_id == current_user.id,
        )
    ).scalars().first()

    if record is None:
        if is_ajax():
            return jsonify(ok=False, error="Transfer not found."), 404
        abort(404)

    candidate_files = {
        record.content_filename, record.style_filename, record.output_filename
    }
    try:
        db.session.delete(record)
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        logger.exception("Could not delete transfer %s", transfer_id)
        if is_ajax():
            return jsonify(ok=False, error="Could not delete. Try again."), 500
        flash("Could not delete that transfer. Try again.", "danger")
        return redirect(url_for("history"))

    # Delete files only AFTER the DB commit succeeded, and never a file another
    # of this user's history rows still uses (inputs can be reused).
    remaining = db.session.execute(
        db.select(
            StyleTransfer.content_filename,
            StyleTransfer.style_filename,
            StyleTransfer.output_filename,
        ).where(StyleTransfer.user_id == current_user.id)
    ).all()
    still_used = {name for row in remaining for name in row}
    for name in candidate_files - still_used:
        path = existing_user_image(name)
        if path is not None:
            path.unlink(missing_ok=True)

    if is_ajax():
        return jsonify(ok=True, remaining=len(remaining))
    flash("Transfer deleted.", "success")
    return redirect(url_for("history"))


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(BASE_DIR / "examples", filename)


# --------------------------------------------------
# Database initialisation
# --------------------------------------------------
# FIX: db.create_all() ran unconditionally at import. With gunicorn -w N every
# worker raced to create the same tables (duplicate-table errors), and an
# unreachable DB crashed the import with no clear message. It is now opt-out,
# wrapped, and meant for first-run/dev only; use `flask db upgrade` in production.
if os.environ.get("AUTO_CREATE_DB", "1") == "1":
    with app.app_context():
        try:
            db.create_all()
        except IntegrityError:
            db.session.rollback()  # another worker created the tables first
        except SQLAlchemyError:
            logger.exception(
                "Could not connect to the database. Check DATABASE_URL."
            )
            raise

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )