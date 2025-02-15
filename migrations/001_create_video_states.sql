CREATE TABLE IF NOT EXISTS video_states (
    video_id UUID PRIMARY KEY,
    video_path TEXT NOT NULL,
    original_name TEXT NOT NULL,
    faces JSONB,
    expressions JSONB,
    activities JSONB,
    summary TEXT,
    total_frames INTEGER DEFAULT 0,
    anomalies INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
