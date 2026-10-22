import hashlib
import json
import logging
import os
import re
import shutil
import smtplib
import ssl
import threading
import uuid
from email.message import EmailMessage
from datetime import datetime, timezone
from pathlib import Path

import torch
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, flash, abort, jsonify
)
from flask_bootstrap import Bootstrap
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import (
    LoginManager, UserMixin, login_user,
    logout_user, login_required, current_user
)
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from PIL import Image
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from torchvision import transforms
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from wtforms import (
    FileField, SubmitField, FloatField, HiddenField,
    StringField, PasswordField
)
from wtforms.validators import DataRequired, EqualTo, Length

from utils.utils import adaptive_instance_normalizaiton
from utils.models import VGGEncoder, Decoder

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
if not app.config["SECRET_KEY"]:
    raise RuntimeError("Set SECRET_KEY in your .env file.")

csrf = CSRFProtect(app)

# Behind Nginx / Caddy / Cloudflare Tunnel every request would otherwise look like
# it comes from the proxy's IP, so rate limits would hit all users at once.
# Set TRUSTED_PROXY=1 only when you really are behind exactly one proxy.
if os.environ.get("TRUSTED_PROXY", "0") == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Password reset
app.config["RESET_TOKEN_MAX_AGE"] = int(os.environ.get("RESET_TOKEN_MAX_AGE", "3600"))  # seconds
APP_BASE_URL = os.environ.get("APP_BASE_URL", "").rstrip("/")  # e.g. https://styleforge.example.com

# Rate limiting (in-memory by default; use Redis when running several workers/servers:
# RATELIMIT_STORAGE_URI=redis://localhost:6379)
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
    default_limits=[],
)


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
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{Path(app.instance_path) / 'app.db'}"
    logger.warning("DATABASE_URL not set; using local SQLite database.")

database_url = normalise_database_url(database_url)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
if not database_url.startswith("sqlite"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 280,
        "pool_size": 5,
        "max_overflow": 5,
    }

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["WTF_CSRF_TIME_LIMIT"] = None
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
Path(app.instance_path).mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT = Path(app.instance_path) / "user_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

db = SQLAlchemy()
db.init_app(app)
migrate = Migrate(app, db)  
Bootstrap(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Please log in to use style transfer."
login_manager.init_app(app)
class User(UserMixin, db.Model):
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
        index=True,
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


@app.errorhandler(429)
def handle_rate_limit(exc):
    message = "Too many attempts. Please wait a few minutes and try again."
    if is_ajax():
        return jsonify(ok=False, error=message), 429
    flash(message, "danger")
    return redirect(url_for("index" if current_user.is_authenticated else "login"))


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
class RegistrationForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=80)])
    email = StringField("Email", validators=[DataRequired(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=8, max=128)])
    submit = SubmitField("Create account")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Length(max=255)])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Log in")


class ForgotPasswordForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Length(max=255)])
    submit = SubmitField("Send reset link")


class ResetPasswordForm(FlaskForm):
    password = PasswordField("New password", validators=[DataRequired(), Length(min=8, max=128)])
    confirm = PasswordField(
        "Confirm new password",
        validators=[DataRequired(), EqualTo("password", message="Passwords do not match.")],
    )
    submit = SubmitField("Reset password")


class UploadForm(FlaskForm):
    content = FileField("Content_Image")
    style = FileField("Style_Image")
    content_path = HiddenField()
    style_path = HiddenField()
    style_id = HiddenField()   # id of a preset from the style library (static/styles)
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")
@app.route("/register", methods=["GET", "POST"])
@limiter.limit("10 per hour", methods=["POST"])
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
        ).scalars().first()
        if existing:
            flash("That username or email is already registered.", "danger")
            return render_template("register.html", form=form)

        user = User(username=username, email=email)
        user.set_password(form.password.data)
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
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
@limiter.limit("10 per minute; 50 per hour", methods=["POST"])
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


# --------------------------------------------------
# Password reset (forgot password)
# --------------------------------------------------
def _reset_serializer():
    return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="password-reset-v1")


def _password_fingerprint(user):
    # Part of the token: once the password changes the fingerprint changes, so
    # a reset link can be used exactly once and old links die automatically.
    return hashlib.sha256(user.password_hash.encode("utf-8")).hexdigest()[:20]


def make_reset_token(user):
    return _reset_serializer().dumps({"uid": user.id, "fp": _password_fingerprint(user)})


def read_reset_token(token):
    """Return (user, None) on success or (None, 'expired' | 'invalid')."""
    try:
        data = _reset_serializer().loads(token, max_age=app.config["RESET_TOKEN_MAX_AGE"])
    except SignatureExpired:
        return None, "expired"
    except BadSignature:
        return None, "invalid"
    user = db.session.get(User, data.get("uid")) if isinstance(data, dict) else None
    if user is None or data.get("fp") != _password_fingerprint(user):
        return None, "invalid"
    return user, None


def external_url(endpoint, **values):
    """Absolute URL for e-mails. APP_BASE_URL avoids trusting the Host header."""
    if APP_BASE_URL:
        return APP_BASE_URL + url_for(endpoint, **values)
    return url_for(endpoint, _external=True, **values)


def _send_smtp(message):
    server = os.environ["MAIL_SERVER"]
    port = int(os.environ.get("MAIL_PORT", "587"))
    username = os.environ.get("MAIL_USERNAME")
    password = os.environ.get("MAIL_PASSWORD")
    if os.environ.get("MAIL_USE_SSL", "0") == "1":
        smtp = smtplib.SMTP_SSL(server, port, timeout=20, context=ssl.create_default_context())
    else:
        smtp = smtplib.SMTP(server, port, timeout=20)
    with smtp:
        if os.environ.get("MAIL_USE_SSL", "0") != "1" and os.environ.get("MAIL_USE_TLS", "1") == "1":
            smtp.starttls(context=ssl.create_default_context())
        if username:
            smtp.login(username, password or "")
        smtp.send_message(message)


def send_email(to_address, subject, text_body, html_body=None):
    """Send in a background thread so the response time never reveals whether
    an account exists. Without MAIL_SERVER the message is printed to the log
    (development mode)."""
    if not os.environ.get("MAIL_SERVER"):
        logger.warning("MAIL_SERVER not set - e-mail NOT sent. To: %s | %s\n%s",
                       to_address, subject, text_body)
        return

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = os.environ.get("MAIL_FROM") or os.environ.get("MAIL_USERNAME") or "no-reply@localhost"
    message["To"] = to_address
    message.set_content(text_body)
    if html_body:
        message.add_alternative(html_body, subtype="html")

    def worker():
        try:
            _send_smtp(message)
        except Exception:
            logger.exception("Could not send e-mail to %s", to_address)

    threading.Thread(target=worker, daemon=True).start()


def send_reset_email(user):
    link = external_url("reset_password", token=make_reset_token(user))
    minutes = app.config["RESET_TOKEN_MAX_AGE"] // 60
    text = (
        f"Hi {user.username},\n\n"
        f"We received a request to reset your StyleForge AI password.\n"
        f"Open this link to choose a new one (valid for {minutes} minutes):\n\n{link}\n\n"
        f"If you didn't ask for this, you can ignore this e-mail; your password stays the same."
    )
    html = (
        f"<p>Hi {user.username},</p>"
        f"<p>We received a request to reset your StyleForge AI password.</p>"
        f'<p><a href="{link}" style="background:#6366f1;color:#fff;padding:12px 22px;'
        f'border-radius:10px;text-decoration:none;display:inline-block">Choose a new password</a></p>'
        f"<p>The link is valid for {minutes} minutes and works once.</p>"
        f"<p style='color:#666'>If you didn't ask for this, ignore this e-mail; your password stays the same.</p>"
    )
    send_email(user.email, "Reset your StyleForge AI password", text, html)


def send_password_changed_email(user):
    send_email(
        user.email,
        "Your StyleForge AI password was changed",
        f"Hi {user.username},\n\nYour password was just changed. If this wasn't you, "
        f"reset it again right away and contact support.",
    )


@app.route("/forgot-password", methods=["GET", "POST"])
@limiter.limit("5 per hour; 15 per day", methods=["POST"])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = form.email.data.strip().lower()
        user = db.session.execute(
            db.select(User).where(db.func.lower(User.email) == email)
        ).scalars().first()
        if user is not None:
            try:
                send_reset_email(user)
            except Exception:
                logger.exception("Could not build reset e-mail")
        # Same answer whether or not the account exists (no account enumeration).
        flash("If an account exists for that e-mail, we've sent a link to reset the password.", "success")
        return redirect(url_for("login"))
    return render_template("forgot_password.html", form=form)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
@limiter.limit("10 per hour", methods=["POST"])
def reset_password(token):
    user, problem = read_reset_token(token)
    if user is None:
        flash(
            "That reset link has expired. Request a new one." if problem == "expired"
            else "That reset link is invalid or was already used. Request a new one.",
            "danger",
        )
        return redirect(url_for("forgot_password"))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        try:
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            logger.exception("Password reset failed")
            flash("Could not update the password right now. Please try again.", "danger")
            return render_template("reset_password.html", form=form)
        send_password_changed_email(user)
        flash("Password updated. You can log in now.", "success")
        return redirect(url_for("login"))
    return render_template("reset_password.html", form=form)


# --------------------------------------------------
# Style library (preset styles users can pick)
# --------------------------------------------------
STYLES_DIR = BASE_DIR / "static" / "styles"
STYLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,60}$")
STYLE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_style_library():
    """Read static/styles/styles.json; any image in that folder that is not listed
    is added automatically so people can just drop a file in."""
    items, seen = [], set()
    manifest = STYLES_DIR / "styles.json"
    if manifest.is_file():
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.exception("Could not read %s", manifest)
            entries = []
        for e in entries:
            sid, fname = str(e.get("id", "")), str(e.get("file", ""))
            if not STYLE_ID_RE.match(sid) or sid in seen or Path(fname).name != fname:
                continue
            if not (STYLES_DIR / fname).is_file():
                logger.warning("Style %s listed but %s is missing", sid, fname)
                continue
            seen.add(sid)
            items.append({
                "id": sid, "file": fname,
                "name": str(e.get("name") or sid.replace("-", " ").title()),
                "category": str(e.get("category") or "More"),
                "description": str(e.get("description") or ""),
            })
    listed_files = {i["file"] for i in items}
    if STYLES_DIR.is_dir():
        for path in sorted(STYLES_DIR.iterdir()):
            sid = re.sub(r"[^a-z0-9_-]+", "-", path.stem.lower()).strip("-")
            if (path.suffix.lower() in STYLE_EXTENSIONS and path.name not in listed_files
                    and STYLE_ID_RE.match(sid) and sid not in seen):
                seen.add(sid)
                items.append({"id": sid, "file": path.name, "name": path.stem.replace("-", " ").replace("_", " ").title(),
                              "category": "More", "description": ""})
    for i in items:
        thumb = STYLES_DIR / "thumbs" / i["file"]
        i["thumb"] = f"styles/thumbs/{i['file']}" if thumb.is_file() else f"styles/{i['file']}"
        i["image"] = f"styles/{i['file']}"
    return items


STYLE_LIBRARY = load_style_library()
STYLE_BY_ID = {s["id"]: s for s in STYLE_LIBRARY}
STYLE_CATEGORIES = list(dict.fromkeys(s["category"] for s in STYLE_LIBRARY))
PRESET_NAME_RE = re.compile(r"^preset_([a-z0-9][a-z0-9_-]*)\.(?:jpg|jpeg|png)$")


def copy_preset_to_user(style):
    """Copy a preset into the user's own folder (once) so history, reuse and delete
    work exactly like they do for uploaded styles, with no database change."""
    suffix = Path(style["file"]).suffix.lower()
    name = f"preset_{style['id']}{suffix}"
    destination = user_upload_dir() / name
    if not destination.is_file():
        shutil.copyfile(STYLES_DIR / style["file"], destination)
    return name


def preset_label(filename):
    match = PRESET_NAME_RE.match(filename or "")
    style = STYLE_BY_ID.get(match.group(1)) if match else None
    return style["name"] if style else None


@app.context_processor
def inject_style_library():
    return {"style_library": STYLE_LIBRARY, "style_categories": STYLE_CATEGORIES}



@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have logged out.", "success")
    return redirect(url_for("login"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VGGEncoder(str(BASE_DIR / "vgg_normalised.pth")).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(
    torch.load(
        BASE_DIR / "experiment" / "experiment1" / "decoder_128.pth",
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
    preset = None

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

            preset = None
            if form.style.data and form.style.data.filename:
                style_filename = save_uploaded_file(form.style.data)       # 1) uploaded file wins
            elif (form.style_id.data or "").strip():                         # 2) preset from the library
                preset = STYLE_BY_ID.get(form.style_id.data.strip())
                if preset is None:
                    raise ValueError("That style is not available. Pick another one.")
                style_filename = copy_preset_to_user(preset)
            else:                                                            # 3) re-use earlier style
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
            db.session.rollback()
            error = str(exc)

        except SQLAlchemyError:
            db.session.rollback()
            logger.exception("Could not save style-transfer history")
            if result_path is not None and result_path.exists():
                result_path.unlink(missing_ok=True)
            error = "Could not save your result to the database. Please try again."

        except Exception:
            db.session.rollback()
            logger.exception("Style transfer failed")
            error = "Style transfer failed. Please try different images."
    if content_filename:
        form.content_path.data = content_filename
    if style_filename:
        form.style_path.data = style_filename
    if preset is not None:
        form.style_id.data = preset["id"]
    elif style_filename:
        form.style_id.data = ""

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
            style_id=preset["id"] if preset else None,
            style_name=preset["name"] if preset else None,
            result_url=img_url(result_image),
        ), (200 if result_image else 400)

    initial = {
        "result_url": url_for("send_image", filename=result_image) if result_image else None,
        "content_url": url_for("send_image", filename=content_filename) if result_image and content_filename else None,
        "alpha": alpha if result_image else None,
        "style_name": preset["name"] if preset else None,
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
    pagination = db.paginate(
        db.select(StyleTransfer)
        .where(StyleTransfer.user_id == current_user.id)
        .order_by(StyleTransfer.created_at.desc()),
        page=page, per_page=20, error_out=False,
    )
    transfers = pagination.items
    for t in transfers:
        t.style_label = preset_label(t.style_filename)
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
if os.environ.get("AUTO_CREATE_DB", "1") == "1":
    with app.app_context():
        try:
            db.create_all()
        except IntegrityError:
            db.session.rollback() 
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