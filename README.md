# Tech Challenge - Pipeline de Análise de Vídeo com FastAPI e n8n

## Descrição

Esta solução implementa uma aplicação em que o processamento de um vídeo é dividido em cinco etapas:
1. **/video_upload**: Recebe o vídeo, salva em disco e dispara o workflow no n8n.
2. **/face_recognition**: Detecta e marca os rostos do vídeo.
3. **/expression_analisys**: Analisa (simuladamente) as expressões emocionais dos rostos.
4. **/ativity_detection**: Detecta e categoriza atividades (simulando anomalias).
5. **/resume**: Agrega os resultados e gera um resumo.

O fluxo é orquestrado via n8n, que ao final envia um e-mail com o resumo do processamento.

## Estrutura do Projeto

```
tech-challenge/ 
├── docker-compose.yml 
├── README.md 
├── n8n_workflow.json 
└── fastapi_service/ 
    ├── Dockerfile 
    ├── requirements.txt 
    └── main.py
```

## Pré-requisitos

- Docker e Docker Compose instalados na máquina.

## Instruções para Execução

1. **Clone o repositório** e acesse a pasta do projeto:
   ```bash
   git clone 
   cd techchallenge4
2. Construa e inicie os containers:
    ```
    docker-compose up --build
    ```
3. Execute a Migration:
   
   Após os containers estarem ativos, execute o script de migration para criar a tabela necessária:
   ```
   docker exec -i postgres psql -U postgres -d video_db < migrations/001_create_video_states.sql
   ```
4. Acesse a API do FastAPI:
    
    A API estará disponível em http://localhost:8000/docs (documentação interativa gerada pelo FastAPI).
5. Configure o n8n:
   - Acesse a interface do n8n em http://localhost:5678 e clique em "Start from Scratch".
   - Importe o arquivo n8n_workflow.json (Menu [...] → Import from File → Selecione o arquivo `n8n_workflow.json`).
   - Ajuste os parâmetros do nó de envio de e-mail (como fromEmail e toEmail) conforme sua configuração.

6. Teste o fluxo:
   - Envie um vídeo utilizando o endpoint /video_upload (por exemplo, via Swagger ou cURL).
   - O FastAPI salvará o vídeo, disparará o webhook no n8n e o fluxo será executado, chamando sequencialmente os endpoints de processamento.
   - Ao final, o n8n enviará um e-mail com o resumo do processamento.
