import io
import cv2
import pyodbc
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from base_models import User, UserCreate

# Load models in a separate module or use a more robust model management system
model = tf.keras.models.load_model("model_over600.h5", compile=False)
model_CNN = tf.keras.models.load_model("MY_MODEL.h5", compile=False)

conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL '
                      'Server};SERVER=SQL8006.site4now.net;DATABASE=db_aa9a89_criminaldetection;UID'
                      '=db_aa9a89_criminaldetection_admin;PWD=Ae105222')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

origins = [
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_image_as_bytes(file: UploadFile) -> bytes:
    return file.read()


def show_image_from_db(db_connection, row_id: int) -> io.BytesIO:
    cursor = db_connection.cursor()
    query = "SELECT image_RGB FROM dbo.image_RGB WHERE id = ?"
    cursor.execute(query, (row_id + 1,))
    image_data = cursor.fetchone()
    if image_data is None:
        raise HTTPException(status_code=404, detail="Image not found")
    image_stream = io.BytesIO(image_data[0])
    return image_stream


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    img_io = io.BytesIO(img_bytes)
    img_pil = Image.open(img_io)
    img_np = np.array(img_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    resized_img = cv2.resize(img_cv2, (256, 256))
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype('float32') / 255.0
    normalized_img = np.expand_dims(normalized_img, axis=0)
    return normalized_img


def predict_image(normalized_img: np.ndarray) -> int:
    predictions = model.predict(normalized_img)
    output_image = Image.fromarray((predictions[0] * 255).astype(np.uint8))
    output_array = np.array(output_image)
    resized_img = cv2.resize(output_array, (256, 256))
    img = np.expand_dims(resized_img, axis=0)
    prediction_img = model_CNN.predict(img)
    oi = Image.fromarray((prediction_img[0] * 255).astype(np.uint8))
    predicted_class = np.argmax(oi)
    return predicted_class


# Login
@app.post("/login")
async def login(email: str, password: str):
    cursor = conn.cursor()
    query = "SELECT id, name, phone, email, pass FROM dbo.users WHERE email =? AND pass =?"
    cursor.execute(query, (email, password))
    user = cursor.fetchone()
    if user:
        # return {"id": user[0], "name": user[1], "phone": user[2], "email": user[3], "password": user[4]}
        return {"Successful Login"}
    else:
        raise HTTPException(status_code=401, detail="Invalid email or password")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await read_image_as_bytes(file)
        normalized_img = preprocess_image(img_bytes)
        predicted_class = predict_image(normalized_img)
        img_data = show_image_from_db(conn, int(predicted_class) - 1)
        return StreamingResponse(content=img_data, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/", response_model=list[User])
async def get_users():
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, phone, email, pass FROM dbo.users")
    rows = cursor.fetchall()
    users = [User(id=row[0], name=row[1], phone=row[2], email=row[3], password=row[4]) for row in rows]
    return users


# Get a user by ID
@app.get("/users/{user_id}", response_model=User)
def get_user(user_id: int):
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, phone, email, pass FROM dbo.users WHERE id = ?", str(user_id))
    user = cursor.fetchone()
    if user:
        return user
    else:
        raise HTTPException(status_code=404, detail="User not found")


# Create a new user
@app.post("/users", response_model=User)
def create_user(user: UserCreate):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO dbo.users (name, phone, email, pass) VALUES (?, ?, ?, ?)",
                   (user.name, user.phone, user.email, user.password))
    conn.commit()
    cursor.execute("SELECT id, name, phone, email, pass FROM dbo.users WHERE id = ?", str(user.id))
    return {"id": user.id, "name": user.name, "phone": user.phone, "email": user.email, "password": user.password}


# Update a user
@app.put("/users/{user_id}", response_model=User)
def update_user(user_id: int, user: User):
    cursor = conn.cursor()
    cursor.execute("UPDATE dbo.users SET name = ?, phone = ?, email = ?, pass = ? WHERE id = ?",
                   (user.name, user.phone, user.email, user.password, str(user_id)))
    conn.commit()
    cursor.execute("SELECT id, name, phone, email, pass FROM dbo.users WHERE id = ?", str(user_id))
    updated_user = cursor.fetchone()
    return updated_user


# Delete a user
@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    cursor = conn.cursor()
    # Delete related records in the image_RGB table
    cursor.execute("DELETE FROM dbo.image_RGB WHERE ske_id IN (SELECT id FROM dbo.image_sketch WHERE user_id = ?)",
                   str(user_id))
    conn.commit()
    cursor.execute("DELETE FROM dbo.image_sketch WHERE user_id = ?", str(user_id))
    conn.commit()
    cursor.execute("DELETE  FROM dbo.users WHERE id = ?", str(user_id))
    conn.commit()
    return {"detail": "User deleted"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
