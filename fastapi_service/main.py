import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
from collections import Counter
from datetime import datetime

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import torch
import torch.nn.functional as torch_functional
import torchvision
import torchvision.transforms as transforms
import uuid
import requests
import psycopg2
import json
from deepface import DeepFace

app = FastAPI(title="Tech Challenge - Pipeline de Análise de Vídeo")

if not os.path.exists("videos"):
    os.makedirs("videos")


def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="video_db",
        user="postgres",
        password="postgres"
    )
    return conn


cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_activity = torchvision.models.video.r3d_18(pretrained=True).to(device)
model_activity.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])

@app.post("/video_upload")
async def video_upload(file: UploadFile = File(...)):
    """
    Recebe o arquivo de vídeo, salva em disco, insere o estado inicial no banco
    e dispara o webhook do n8n para iniciar o fluxo.
    """
    video_id = str(uuid.uuid4())
    original_name = file.filename
    file_path = f"videos/{video_id}.mp4"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    conn = get_db_connection()
    cur = conn.cursor()
    insert_sql = """
        INSERT INTO video_states (video_id, video_path, original_name)
        VALUES (%s, %s, %s)
    """
    cur.execute(insert_sql, (video_id, file_path, original_name))
    conn.commit()
    cur.close()
    conn.close()

    try:
        response = requests.post("http://localhost:5678/webhook/video_process", json={"video_id": video_id})
        print("response n8n:", response.text)
        if response.status_code != 200:
            return {"error": "Falha ao iniciar o workflow n8n", "status_code": response.status_code, "details": response.text}
    except Exception as e:
        return {"error": "Falha ao iniciar o workflow n8n", "details": str(e)}
    return {"message": "Vídeo recebido e processo iniciado", "video_id": video_id}


@app.get("/face_recognition")
def face_recognition(video_id: str):
    """
    Realiza o reconhecimento facial, identificando e marcando os rostos do vídeo.
    Atualiza o campo 'faces' na tabela video_states.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT video_path FROM video_states WHERE video_id = %s", (video_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return {"error": "video_id não encontrado"}
    video_path = row[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cur.close()
        conn.close()
        return {"error": "Não foi possível abrir o vídeo"}

    face_results = []
    frame_interval = 30
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            faces_info = []
            for (x, y, w, h) in faces:
                faces_info.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
            face_results.append({"frame": frame_idx, "faces": faces_info})
        frame_idx += 1
    cap.release()

    update_sql = "UPDATE video_states SET faces = %s WHERE video_id = %s"
    cur.execute(update_sql, (json.dumps(face_results), video_id))
    conn.commit()
    cur.close()
    conn.close()

    return {"video_id": video_id}


@app.get("/expression_analisys")
def expression_analisys(video_id: str):
    """
    Realiza a análise das expressões emocionais com base nos rostos detectados.
    Em vez de selecionar emoções aleatoriamente, utiliza a biblioteca DeepFace para
    uma avaliação real da emoção dominante de cada rosto detectado.
    Atualiza o campo 'expressions' na tabela video_states.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT video_path, faces FROM video_states WHERE video_id = %s", (video_id,))
    row = cur.fetchone()
    if not row or row[0] is None or row[1] is None:
        cur.close()
        conn.close()
        return {"error": "Reconhecimento facial não realizado ou video_id não encontrado"}

    video_path, faces_json = row[0], row[1]
    faces = faces_json

    expression_results = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cur.close()
        conn.close()
        return {"error": "Não foi possível abrir o vídeo"}

    for item in faces:
        frame_number = item["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue
        for face in item["faces"]:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            face_img = frame[y:y+h, x:x+w]
            try:
                analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0].get('dominant_emotion', 'undefined')
            except Exception as e:
                print("error ", e)
                emotion = "erro"
            expression_results.append({
                "frame": frame_number,
                "face": face,
                "emotion": emotion
            })

    cap.release()

    update_sql = "UPDATE video_states SET expressions = %s WHERE video_id = %s"
    cur.execute(update_sql, (json.dumps(expression_results), video_id))
    conn.commit()
    cur.close()
    conn.close()

    return {"video_id": video_id, "expression_analysis": expression_results}


@app.get("/activity_detection")
def activity_detection(video_id: str):
    """
    Detecta e categoriza as atividades realizadas no vídeo utilizando um modelo
    de reconhecimento de ações (r3d_18). O vídeo é dividido em clipes de 16 frames,
    cada clipe é analisado e a atividade predita é registrada.

    Atualiza os campos 'activities', 'total_frames' e 'anomalies' na tabela video_states.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT video_path FROM video_states WHERE video_id = %s", (video_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return JSONResponse(status_code=404, content={"error": "video_id não encontrado"})
    video_path = row[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cur.close()
        conn.close()
        return {"error": "Não foi possível abrir o vídeo"}

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    clip_length = 16
    num_segments = total_frames // clip_length
    if num_segments == 0:
        cur.close()
        conn.close()
        return {"error": "Vídeo muito curto para segmentação de atividade"}

    activity_results = []
    anomalies = 0
    THRESHOLD_CONFIDENCE = 0.6

    for i in range(num_segments):
        clip_frames = frames[i * clip_length: (i + 1) * clip_length]
        processed_frames = []
        for frame in clip_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = preprocess(frame_rgb)
            processed_frames.append(tensor_frame)
        clip_tensor = torch.stack(processed_frames)
        clip_tensor = clip_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model_activity(clip_tensor)
            probs = torch_functional.softmax(outputs, dim=1)
            max_prob, predicted_index = torch.max(probs, dim=1)

        if max_prob.item() < THRESHOLD_CONFIDENCE:
            activity_label = "anomalia"
            anomalies += 1
        else:
            activity_label = get_activity_label(predicted_index.item())

        activity_results.append({
            "segment": i,
            "start_frame": i * clip_length,
            "activity": activity_label
        })

    update_sql = """
        UPDATE video_states 
        SET activities = %s, total_frames = %s, anomalies = %s 
        WHERE video_id = %s
    """
    cur.execute(update_sql, (json.dumps(activity_results), total_frames, anomalies, video_id))
    conn.commit()
    cur.close()
    conn.close()

    return {
        "video_id": video_id,
        "activity_detection": activity_results,
        "total_frames": total_frames,
        "anomalies": anomalies
    }


@app.get("/resume")
def resume(video_id: str):
    """
    Agrega os resultados das etapas anteriores e gera um resumo formal do processamento.
    O resumo conterá:
    - Principais emoções detectadas
    - Principais atividades detectadas
    - Anomalias detectadas (caso existam)

    O campo summary será um texto formal em português que será enviado por email.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT total_frames, anomalies, expressions, activities, original_name, created_at 
        FROM video_states 
        WHERE video_id = %s
    """, (video_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return {"error": "video_id não encontrado"}

    total_frames, anomalies_count, expressions, activities, original_name, created_at = row

    if expressions:
        expressions_list = expressions
        emotions = [item["emotion"] for item in expressions_list if "emotion" in item]
        counter_emotions = Counter(emotions)
        principais_emocoes = ", ".join([emotion for emotion, _ in counter_emotions.most_common()])
    else:
        principais_emocoes = "Nenhuma emoção detectada"

    if activities:
        activities_list = activities
        activities_labels = [item["activity"] for item in activities_list if "activity" in item]
        counter_activities = Counter(activities_labels)
        principais_atividades = ", ".join([activity for activity, _ in counter_activities.most_common()])

        anomalias_segmentos = [f"segmento {item['segment']}" for item in activities_list if item.get("activity", "").lower() == "anomalia"]
        if anomalias_segmentos:
            anomalias_str = ", ".join(anomalias_segmentos)
        else:
            anomalias_str = "Nenhuma anomalia detectada"
    else:
        principais_atividades = "Nenhuma atividade detectada"
        anomalias_str = "Nenhuma anomalia detectada"

    data_hora = created_at.strftime("%d/%m/%Y %H:%M:%S") if isinstance(created_at, datetime) else str(created_at)

    summary_text = (
        f"O processamento do vídeo '{original_name}' recebido em {data_hora} foi realizado, segue resumo:\n\n"
        f"Principais emoções: {principais_emocoes}\n\n"
        f"Principais atividades: {principais_atividades}\n\n"
        f"Anomalias detectadas: {anomalias_str}"
    )

    update_sql = "UPDATE video_states SET summary = %s WHERE video_id = %s"
    cur.execute(update_sql, (summary_text, video_id))
    conn.commit()
    cur.close()
    conn.close()

    return {
        "video_id": video_id,
        "summary": summary_text,
        "principais_emocoes": principais_emocoes,
        "principais_atividades": principais_atividades,
        "anomalias": anomalias_str
    }


@app.get("/annotate_video")
def annotate_video(video_id: str):
    """
    Lê o vídeo original e adiciona as anotações visuais com base nos dados processados:
      - Desenha um quadrado vermelho em volta dos rostos detectados. O quadrado permanece no lugar até que uma nova detecção seja registrada.
      - Exibe, abaixo de cada rosto, a expressão detectada.
      - Exibe, no canto inferior esquerdo, o nome da atividade atual.

    O vídeo anotado é salvo na pasta "processed_videos".
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT video_path, faces, expressions, activities, original_name 
        FROM video_states 
        WHERE video_id = %s
    """, (video_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return {"error": "video_id não encontrado"}
    video_path, faces, expressions, activities, original_name = row
    cur.close()
    conn.close()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Não foi possível abrir o vídeo original"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if not os.path.exists("processed_videos"):
        os.makedirs("processed_videos")
    output_path = f"processed_videos/processed_{video_id}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    faces_sorted = sorted(faces, key=lambda x: x["frame"]) if faces else []
    activities_sorted = sorted(activities, key=lambda x: x["start_frame"]) if activities else []

    next_face_idx = 0
    next_activity_idx = 0
    current_faces = []
    current_activity = ""

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if next_face_idx < len(faces_sorted) and frame_num >= faces_sorted[next_face_idx]["frame"]:
            current_faces = faces_sorted[next_face_idx]["faces"]
            next_face_idx += 1

        if activities_sorted:
            while next_activity_idx < len(activities_sorted) and frame_num >= activities_sorted[next_activity_idx]["start_frame"]:
                current_activity = activities_sorted[next_activity_idx]["activity"]
                next_activity_idx += 1

        for face in current_faces:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            expr_text = next((item['emotion'] for item in expressions if item['face'] == face), "")

            if expr_text:
                cv2.putText(frame, expr_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if current_activity:
            cv2.putText(frame, current_activity, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()

    return {"message": "Vídeo processado com sucesso", "processed_video": output_path}


def get_activity_label(index):
    """
    Reads a CSV file and returns the `name` corresponding to the given `id`.

    Args:
        :param index: The ID for which the corresponding name is to be retrieved.

    Returns:
        str: The name corresponding to the given ID, or None if the ID is not found.

    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'kinect-400.csv')

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['id']) == index:
                return row['name']
    return None


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
