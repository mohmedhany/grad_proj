import io
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import cv2
import numpy as np
from PIL import Image
from starlette.responses import StreamingResponse
import pyodbc
from typing import List

from base_models import User, UserCreate
from model import model_load

conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=DESKTOP-IOILHI2;DATABASE=FaceCriminalDetection'
                      ';Trusted_Connection=yes;')


model = model_load()

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_io = io.BytesIO(img_bytes)
    img_pil = Image.open(img_io)
    img_np = np.array(img_pil)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    resized_img = cv2.resize(img_cv2, (256, 256))
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    normalized_img = rgb_img.astype('float32') / 255.0

    if len(normalized_img.shape) == 3:  # Check if it has 3 channels (RGB)
        normalized_img = np.expand_dims(normalized_img, axis=0)

    predictions = model.predict(normalized_img)
    output_image = Image.fromarray((predictions[0] * 255).astype(np.uint8))
    output_bytes = io.BytesIO()
    output_image.save(output_bytes, format='JPEG')
    output_bytes.seek(0)
    return StreamingResponse(content=output_bytes, media_type="image/jpeg")


@app.get("/users", response_model=List[User])
def get_users():
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, phone, email, password FROM dbo.users")
    users = cursor.fetchall()
    return users


# Get a user by ID
@app.get("/users/{user_id}", response_model=User)
def get_user(user_id: int):
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, phone, email, password FROM dbo.users WHERE id = ?", str(user_id))
    user = cursor.fetchone()
    if user:
        return user
    else:
        raise HTTPException(status_code=404, detail="User not found")


# Create a new user
@app.post("/users", response_model=User)
def create_user(user: UserCreate):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO dbo.users (name, phone, email, password) VALUES (?, ?, ?, ?)",
                   (user.name, user.phone, user.email, user.password))
    conn.commit()
    cursor.execute("SELECT id, name, phone, email, password FROM dbo.users WHERE id = ?", str(user.id))
    return {"id": user.id, "name": user.name, "phone": user.phone, "email": user.email, "password": user.password}


# Update a user
@app.put("/users/{user_id}", response_model=User)
def update_user(user_id: int, user: User):
    cursor = conn.cursor()
    cursor.execute("UPDATE dbo.users SET name = ?, phone = ?, email = ?, password = ? WHERE id = ?",
                   (user.name, user.phone, user.email, user.password, str(user_id)))
    conn.commit()
    cursor.execute("SELECT id, name, phone, email, password FROM dbo.users WHERE id = ?", str(user_id))
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
    uvicorn.run(app, port=8000, host="127.0.0.1")
