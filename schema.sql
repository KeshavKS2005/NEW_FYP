-- PostgreSQL Database Schema for 12-Module Open-World Continual Discovery & Retraining Pipeline
-- As defined in fyp_architecture_final_object.md Section 7

CREATE TABLE IF NOT EXISTS unknown_clusters (
    cluster_id SERIAL PRIMARY KEY,
    faiss_index_id INT UNIQUE NOT NULL,
    observation_count INT NOT NULL DEFAULT 1,
    status VARCHAR(32) NOT NULL DEFAULT 'COLLECTING', -- 'COLLECTING', 'IDENTIFIED', 'RETRAINING', 'DEPLOYED_IN_YOLO', 'NOISE'
    identified_label VARCHAR(128) DEFAULT NULL,
    assigned_yolo_class_id INT DEFAULT NULL,
    vlm_confidence FLOAT DEFAULT NULL,
    vlm_reasoning TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS unknown_observations (
    observation_id SERIAL PRIMARY KEY,
    cluster_id INT NOT NULL REFERENCES unknown_clusters(cluster_id) ON DELETE CASCADE,
    track_id INT NOT NULL,
    sub_object_idx INT NOT NULL DEFAULT 0,
    crop_image_path TEXT NOT NULL,
    masked_crop_path TEXT NOT NULL,
    full_frame_path TEXT NOT NULL,
    bbox_xyxy INT[4] NOT NULL,
    frame_hw INT[2] NOT NULL,
    sharpness_score FLOAT NOT NULL,
    cosine_similarity FLOAT NOT NULL,
    decomposition_method VARCHAR(32) NOT NULL,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version_id SERIAL PRIMARY KEY,
    weights_path TEXT NOT NULL,
    num_classes INT NOT NULL,
    class_names_json TEXT NOT NULL,
    old_classes_map50_95 FLOAT NOT NULL,
    new_class_map50 FLOAT NOT NULL,
    new_class_recall FLOAT NOT NULL,
    promoted_to_production BOOLEAN NOT NULL DEFAULT FALSE,
    trained_on_cluster_id INT REFERENCES unknown_clusters(cluster_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cluster_sharpness 
ON unknown_observations(cluster_id, sharpness_score DESC);
