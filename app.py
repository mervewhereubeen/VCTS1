from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import shutil
import os, glob
from fastapi import Request
from starlette.requests import Request



from fastapi.responses import HTMLResponse, FileResponse


from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.orm import sessionmaker, declarative_base

import logging, traceback, hashlib
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T

APP_ORIGIN_UI = ["http://127.0.0.1:5500", "http://localhost:5500"]

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS = DATA_DIR / "uploads"
OUTPUTS = DATA_DIR / "outputs"
DB_PATH = DATA_DIR / "app.db"

for p in (DATA_DIR, UPLOADS, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

app = FastAPI()

from pathlib import Path
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent if 'BASE_DIR' not in globals() else BASE_DIR
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# UI klasörünü servis et
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

# Anasayfa: / -> ui/index3.html
@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("ui/index3.html")


MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "models/weights/best.pt")
WEIGHTS_PATH = Path(MODEL_WEIGHTS_PATH)
MODEL_VERSION = "vcts_model_v1"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")

class PendingCase(Base):
    __tablename__ = "pending_cases"
    case_id = Column(Integer, primary_key=True, autoincrement=True)
    patient_name = Column(String, nullable=False)
    case_created_at = Column(DateTime, default=datetime.utcnow)
    input_path = Column(Text, nullable=False)
    input_url  = Column(Text, nullable=False)
    label = Column(String, nullable=True)
    conf = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    output_path = Column(Text, nullable=True)
    output_url  = Column(Text, nullable=True)

class PublishedCase(Base):
    __tablename__ = "published_cases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, nullable=False)
    patient_name = Column(String, nullable=False)
    case_created_at = Column(DateTime, nullable=False)
    label = Column(String, nullable=True)
    conf = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    doctor_note = Column(Text, nullable=True)
    input_url = Column(Text, nullable=True)
    output_url = Column(Text, nullable=True)
    pred_created_at = Column(DateTime, nullable=True)

Base.metadata.create_all(bind=engine)

class PredictOut(BaseModel):
    case_id: int
    label: Optional[str] = None
    conf: Optional[float] = None
    model_version: Optional[str] = None
    output_url: Optional[str] = None

class PublishIn(BaseModel):
    patient_name: str
    case_id: int
    label: Optional[str] = None
    doctor_note: Optional[str] = None

def url_for_upload(case_id: int, ext: str, request: Request) -> tuple[str, str]:
    up_dir = UPLOADS / str(case_id)
    up_dir.mkdir(parents=True, exist_ok=True)
    fname = f"input{ext}"
    public = f"{str(request.base_url).rstrip('/')}/uploads/{case_id}/{fname}"
    return str(up_dir / fname), public

def url_for_output(case_id: int, request: Request) -> tuple[str, str]:
    out_dir = OUTPUTS / str(case_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "pred.jpg"
    public = f"{str(request.base_url).rstrip('/')}/outputs/{case_id}/{fname}"
    return str(out_dir / fname), public

def _resolve_local_from_url(public_url: str, uploads_root: Path) -> Optional[str]:
    try:
        p = urlparse(public_url).path
        parts = Path(p).parts
        i = parts.index('uploads') if 'uploads' in parts else -1
        if i >= 0:
            rel = Path(*parts[i+1:])
            cand = uploads_root / rel
            return str(cand)
    except Exception:
        pass
    return None

logger = logging.getLogger("uvicorn.error")

MODEL_VERSION = os.getenv("MODEL_VERSION", "oralcls_v2_s384_aug")
WEIGHTS_PATH = Path(os.getenv("MODEL_WEIGHTS_PATH", "models/weights/best.pt"))

IMG_SIZE = 384
CLASS_NAMES = ["benign", "malign"]
DEVICE =  "cpu"
MODEL = None

def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_preprocess():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])

PREPROCESS = build_preprocess()

def load_model_once():
    global MODEL
    if MODEL is not None:
        return

    p = WEIGHTS_PATH
    if not p.exists():
        raise RuntimeError(f"Ağırlık bulunamadı: {p}")

    if p.suffix.lower() in (".pt", ".torchscript", ".ts"):
        try:
            mdl = torch.jit.load(str(p), map_location="cpu")
            mdl.eval()
            MODEL = mdl.to("cpu")
            logger.info(f"[MODEL] TorchScript yüklendi (CPU): {p}")
            return
        except Exception as e:
            logger.warning(f"[MODEL] TorchScript değil/uyumsuz ({e}); checkpoint olarak denenecek.")

    if p.suffix.lower() in (".pt", ".pth"):
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(p), map_location="cpu")

        if hasattr(obj, "state_dict") and callable(getattr(obj, "eval", None)):
            mdl = obj
            mdl.eval()
            MODEL = mdl.to(DEVICE)
            logger.info(f"[MODEL] nn.Module checkpoint yüklendi: {p}")
            return

        state = None
        if isinstance(obj, dict):
            try:
                if obj and all(torch.is_tensor(v) for v in obj.values()):
                    state = obj
            except Exception:
                state = None

            if state is None:
                for k in ["state_dict", "model_state_dict", "net", "weights"]:
                    v = obj.get(k, None)
                    if isinstance(v, dict) and v and all(hasattr(t, "shape") for t in v.values()):
                        state = v
                        logger.info(f"[MODEL] state_dict '{k}' içinde bulundu.")
                        break

            if state is None:
                for k in ["model", "module", "ema"]:
                    v = obj.get(k, None)
                    if hasattr(v, "state_dict"):
                        sd = v.state_dict()
                        state = sd
                        logger.info(f"[MODEL] state_dict {k}.state_dict() ile çıkarıldı.")
                        break

            if state is None:
                for v in obj.values():
                    if isinstance(v, (list, tuple)):
                        for e in v:
                            if hasattr(e, "state_dict"):
                                state = e.state_dict()
                                logger.info("[MODEL] state_dict liste/tuple içindeki module.state_dict() ile çıkarıldı.")
                                break
                        if state is not None:
                            break

        if state is None:
            keys = list(obj.keys()) if isinstance(obj, dict) else type(obj)
            raise RuntimeError(f"Checkpoint içinde state_dict bulunamadı; tip/anahtarlar: {keys}")

        if any(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        mdl = None
        try:
            import timm
            backbone_name = obj.get("arch", "resnet18") if isinstance(obj, dict) else "resnet18"
            mdl = timm.create_model(backbone_name, pretrained=False, num_classes=len(CLASS_NAMES))
            logger.info(f"[MODEL] timm backbone kuruldu: {backbone_name}")
        except Exception:
            import torchvision.models as models
            mdl = models.resnet18(weights=None)
            mdl.fc = torch.nn.Linear(mdl.fc.in_features, len(CLASS_NAMES))
            logger.info("[MODEL] torchvision.resnet18 ile kuruldu (varsayılan).")

        missing, unexpected = mdl.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"[MODEL] missing keys (ilk 8): {list(missing)[:8]}{' ...' if len(missing)>8 else ''}")
        if unexpected:
            logger.warning(f"[MODEL] unexpected keys (ilk 8): {list(unexpected)[:8]}{' ...' if len(unexpected)>8 else ''}")

        mdl.eval()
        MODEL = mdl.to(DEVICE)
        logger.info(f"[MODEL] state_dict yüklendi: {p}")
        return

    if p.suffix.lower() == ".onnx":
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
        MODEL = ort.InferenceSession(str(p), providers=providers)
        logger.info(f"[MODEL] ONNX yüklendi: {p}")
        return

    raise RuntimeError(f"Desteklenmeyen ağırlık uzantısı: {p.suffix}")

def run_model(in_path: str) -> tuple[str, float, np.ndarray | None]:
    load_model_once()

    img = Image.open(in_path).convert("RGB")
    x = PREPROCESS(img).unsqueeze(0)

    if not hasattr(MODEL, "run"):
        x = x.to("cpu")
        with torch.no_grad():
            logits = MODEL(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        conf  = float(probs[idx])
        annotated = np.array(img)
        return label, conf, annotated
    else:
        import onnxruntime as ort
        sess: ort.InferenceSession = MODEL
        inp = x.numpy()
        inp_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        out = sess.run([out_name], {inp_name: inp})[0]
        out = out - out.max(axis=1, keepdims=True)
        probs = np.exp(out) / np.exp(out).sum(axis=1, keepdims=True)
        probs = probs[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]
        conf  = float(probs[idx])
        annotated = np.array(img)
        return label, conf, annotated

@app.on_event("startup")
def _startup_load_model():
    try:
        load_model_once()
    except Exception as e:
        logging.getLogger("uvicorn.error").warning(f"[MODEL] Başlangıçta yüklenemedi: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...), patient_name: str = Form(...)):
    if not patient_name.strip():
        raise HTTPException(400, "patient_name gerekli")

    db = SessionLocal()
    try:
        temp = PendingCase(patient_name=patient_name, input_path="", input_url="")
        db.add(temp); db.commit(); db.refresh(temp)
        case_id = temp.case_id

        ext = Path(file.filename).suffix or ".jpg"
        input_path, input_url = url_for_upload(case_id, ext, request)


        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        temp.input_path = input_path
        temp.input_url = input_url
        db.add(temp); db.commit()

        return {"ok": True, "case_id": case_id, "input_url": input_url}
    finally:
        db.close()

@app.get("/pending")
def get_pending() -> List[dict]:
    db = SessionLocal()
    try:
        rows = db.query(PendingCase).order_by(PendingCase.case_created_at.asc()).all()
        return [
            {
                "case_id": r.case_id,
                "patient_name": r.patient_name,
                "case_created_at": r.case_created_at.isoformat(),
                "input_url": r.input_url,
                "label": r.label,
                "conf": r.conf,
                "model_version": r.model_version,
                "output_url": r.output_url,
            }
            for r in rows
        ]
    finally:
        db.close()

from fastapi.params import Query as QueryParam, Form as FormParam

@app.post("/predict", response_model=PredictOut)
async def predict(
    request: Request,
    case_id: Optional[int] = Query(default=None),
    case_id_form: Optional[int] = Form(default=None),
    payload: Optional[dict] = Body(default=None),
    file: UploadFile | None = File(default=None),
    
):

    if isinstance(case_id, QueryParam):
        case_id = None
    if isinstance(case_id_form, FormParam):
        case_id_form = None

    if case_id is None:
        case_id = case_id_form
    if case_id is None and isinstance(payload, dict):
        try:
            case_id = int(payload.get("case_id"))
        except Exception:
            case_id = None
    if case_id is None:
        raise HTTPException(400, "case_id gerekli")

    db = SessionLocal()
    try:
        it = db.query(PendingCase).filter(PendingCase.case_id == case_id).first()
        if not it:
            raise HTTPException(404, "case bulunamadı")

        in_path = it.input_path
        if not in_path or not os.path.exists(in_path):
            if it.input_url:
                maybe = _resolve_local_from_url(it.input_url, UPLOADS)
                if maybe and os.path.exists(maybe):
                    in_path = maybe
                else:
                    raise HTTPException(500, "input dosyası bulunamadı")
            else:
                raise HTTPException(500, "input dosyası bulunamadı")

        try:
            inp_hash = sha256_of(in_path)
            logger.info(f"[PREDICT] case_id=%s in_path=%s sha256=%s", case_id, in_path, inp_hash[:16])

            label, conf, annotated = run_model(in_path)
            model_version = MODEL_VERSION
        except Exception as e:
            logger.error("MODEL ERROR: %s", e)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"MODEL_ERROR: {e}")

        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        out_dir = OUTPUTS / str(case_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        for pth in glob.glob(str(out_dir / "pred_*.jpg")):
            try:
                os.remove(pth)
            except Exception:
                pass
        out_path = out_dir / f"pred_{ts}.jpg"
        out_url  = f"{str(request.base_url).rstrip('/')}/outputs/{case_id}/pred_{ts}.jpg"

        


        if annotated is None:
            shutil.copy(in_path, out_path)
        else:
            if isinstance(annotated, Image.Image):
                annotated.save(out_path)
            else:
                arr = np.asarray(annotated)
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(arr).save(out_path)

        it.label = label
        it.conf = conf
        it.model_version = model_version
        it.output_path = str(out_path)
        it.output_url  = out_url
        db.add(it)
        db.commit()

        return PredictOut(case_id=case_id, label=label, conf=conf, model_version=model_version, output_url=out_url)
    finally:
        db.close()

@app.post("/cases/{case_id}/predict", response_model=PredictOut)
async def predict_by_path(case_id: int, request: Request):
    return await predict(case_id=case_id, request=request)   

@app.post("/inference", response_model=PredictOut)
async def inference_alias(request: Request, payload: Optional[dict] = Body(default=None)):
    cid = None
    if isinstance(payload, dict):
        try:
            cid = int(payload.get("case_id"))
        except Exception:
            cid = None
    if cid is None:
        raise HTTPException(status_code=400, detail="case_id gerekli")
    return await predict(case_id=cid, request=request) 

@app.post("/publish")
def publish(data: PublishIn):
    db = SessionLocal()
    try:
        it = db.query(PendingCase).filter(PendingCase.case_id == data.case_id).first()
        if not it:
            raise HTTPException(404, "case bulunamadı")

        pub = PublishedCase(
            case_id=it.case_id,
            patient_name=data.patient_name.strip(),
            case_created_at=it.case_created_at,
            label=data.label or it.label,
            conf=it.conf,
            model_version=it.model_version,
            doctor_note=(data.doctor_note or ""),
            input_url=it.input_url,
            output_url=it.output_url,
            pred_created_at=datetime.utcnow() if it.output_url else None,
        )
        db.add(pub)
        db.delete(it)
        db.commit()
        return {"ok": True, "case_id": pub.case_id}
    finally:
        db.close()

@app.get("/cases")
def cases(name: str) -> List[dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(PublishedCase)
            .filter(PublishedCase.patient_name == name.strip())
            .order_by(PublishedCase.case_created_at.asc())
            .all()
        )
        return [
            {
                "case_id": r.case_id,
                "case_created_at": r.case_created_at.isoformat(),
                "label": r.label,
                "conf": r.conf,
                "model_version": r.model_version,
                "doctor_note": r.doctor_note,
                "input_url": r.input_url,
                "output_url": r.output_url,
            }
            for r in rows
        ]
    finally:
        db.close()


# run with:
# python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
