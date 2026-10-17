
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from flask_wtf.csrf import CSRFProtect

import torch
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, flash, abort
)
from flask_bootstrap import Bootstrap
from flask_login import (
    LoginManager, UserMixin, login_user,
    logout_user, login_required, current_user
)
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from werkzeug.security import (
    generate_password_hash, check_password_hash
)
from werkzeug.utils import secure_filename
from wtforms import (
    FileField, SubmitField, FloatField, HiddenField,
    StringField, PasswordField
)
from wtforms.validators import (
    InputRequired, DataRequired, Length
)
from PIL import Image
from torchvision import transforms

from utils.utils import (
    adaptive_instance_normalizaiton, cal_mean_std
)
from utils.models import VGGEncoder, Decoder


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
if not app.config["SECRET_KEY"]:
    raise RuntimeError("Set SECRET_KEY in your .env file.")
csrf = CSRFProtect(app)
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError("Set DATABASE_URL in your .env file.")

# Compatibility with PostgreSQL URLs supplied by some hosts.
if database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://", "postgresql+psycopg://", 1
    )
elif database_url.startswith("postgresql://"):
    database_url = database_url.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Keep uploaded files outside the public static directory.
UPLOAD_ROOT = Path(app.instance_path) / "user_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
Path(app.instance_path).mkdir(parents=True, exist_ok=True)

db = SQLAlchemy()
db.init_app(app)

Bootstrap(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Please log in to use style transfer."
login_manager.init_app(app)


# --------------------------------------------------
# Database models
# --------------------------------------------------

class User(UserMixin, db.Model):
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
        cascade="all, delete-orphan"
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class StyleTransfer(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
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
        nullable=False
    )


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------
# Forms
# --------------------------------------------------

class RegistrationForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=80)]
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Length(max=255)]
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=8, max=128)]
    )
    submit = SubmitField("Create account")


class LoginForm(FlaskForm):
    email = StringField(
        "Email",
        validators=[DataRequired(), Length(max=255)]
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired()]
    )
    submit = SubmitField("Log in")


class UploadForm(FlaskForm):
    content = FileField("Content_Image")
    style = FileField("Style_Image")

    # Preserve your existing form fields.
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
        ).scalar_one_or_none()

        if existing:
            flash("That username or email is already registered.", "danger")
            return render_template("register.html", form=form)

        user = User(username=username, email=email)
        user.set_password(form.password.data)

        db.session.add(user)
        db.session.commit()

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
            db.select(User).where(
                db.func.lower(User.email) == email
            )
        ).scalar_one_or_none()

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
# Existing AdaIN model initialization
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

encoder = VGGEncoder("vgg_normalised.pth").to(device)
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        "experiment/experiment1/decoder_10.pth",
        map_location=device
    )
)

encoder.eval()
decoder.eval()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
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


def style_transfer(
    content_image, style_image, encoder, decoder, alpha, device
):
    # Original AdaIN inference logic preserved.
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feat = encoder(content_image, is_test=True)
        style_feat = encoder(style_image, is_test=True)

        stylized_feat = adaptive_instance_normalizaiton(
            content_feat, style_feat
        )

        stylized_feat = (
            alpha * stylized_feat + (1 - alpha) * content_feat
        )

        stylized_image = decoder(stylized_feat)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone().squeeze(0).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


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

    if form.validate_on_submit():
        try:
            # New uploads take precedence over previously selected images.
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
                content_image, style_image,
                encoder, decoder, alpha, device
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

            result_image = result_filename

        except Exception as exc:
            db.session.rollback()
            error = str(exc)

    return render_template(
        "index.html",
        form=form,
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
    transfers = db.session.execute(
        db.select(StyleTransfer)
        .where(StyleTransfer.user_id == current_user.id)
        .order_by(StyleTransfer.created_at.desc())
    ).scalars().all()

    return render_template("history.html", transfers=transfers)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("examples", filename)


# --------------------------------------------------
# Database initialization
# --------------------------------------------------

with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
