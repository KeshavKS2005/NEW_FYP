"""
Module 9: Automated VLM Class Discovery Trigger (Gemini 2.5 Flash)
Triggers automatic identification when an unknown cluster reaches the observation threshold (100).
Constructs a multimodal prompt from 1 full CCTV scene frame (with red box drawn on an in-memory copy)
and top 3 sharpest crops. Uses Gemini 2.5 Flash Structured Outputs to classify unknown objects.
Pushes confirmed physical objects to Phase 3 retrain_queue.
Adheres strictly to Module 9 in fyp_architecture_final_object.md.
"""

import os
from typing import Optional, List, Any
import cv2
from PIL import Image
from pydantic import BaseModel, Field

from module_9.config import Module9Config
from module_8.db_manager import DatabaseManager


class GeminiObjectSchema(BaseModel):
    """Pydantic structured output contract for Gemini 2.5 Flash."""

    object_name: str = Field(
        ...,
        description="Specific physical object name discovered (e.g. 'bicycle', 'person', 'forklift', 'dog')"
    )
    category: str = Field(
        ...,
        description="High-level category (e.g. 'vehicle', 'animal', 'equipment', 'human', 'environmental_artifact')"
    )
    is_valid_physical_object: bool = Field(
        ...,
        description="True if this is a genuine moving physical object, False if it is noise/shadow/foliage"
    )
    confidence: float = Field(
        ...,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        ...,
        description="Step-by-step analytical reasoning based on contextual scale, shape, and crop appearance"
    )


class VLMDiscoveryIdentifier:
    """
    Automated discovery identification agent using Gemini 2.5 Flash.
    """

    def __init__(
        self,
        config: Optional[Module9Config] = None,
        client: Optional[object] = None
    ):
        self.config = config or Module9Config()
        self._client = client
        if self._client is None and self.config.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.config.api_key)
            except Exception as e:
                print(f"[Module 9] Warning: Could not initialize Google GenAI Client: {e}")
                self._client = None

    def should_trigger(self, observation_count: int, status: str) -> bool:
        """Determines if a cluster should trigger VLM discovery labeling."""
        return observation_count >= self.config.trigger_count and status == "COLLECTING"

    def identify_cluster(
        self,
        cluster_id: int,
        db: DatabaseManager,
        retrain_queue: Optional[Any] = None
    ) -> Optional[GeminiObjectSchema]:
        """
        Executes multimodal VLM query on the top sharpest observations of a cluster.
        Draws the red bounding box ONLY on an in-memory copy of the full frame.

        Args:
            cluster_id: Target cluster ID.
            db: Relational database manager instance.
            retrain_queue: Optional multiprocessing.Queue for Phase 3 continual retraining.

        Returns:
            Parsed GeminiObjectSchema or None if evaluation failed.
        """
        top_rows = db.get_top_observations(
            cluster_id=cluster_id,
            limit=self.config.crop_exemplar_count
        )

        if not top_rows:
            return None

        prompt_text = (
            f"You are a computer vision surveillance analyst. Image 1 shows a full CCTV camera "
            f"frame with a red bounding box highlighting an unknown moving object. Images 2, 3, and 4 "
            f"show close-up crops of this exact same class of object captured across {self.config.trigger_count} "
            f"independent sightings. Identify this specific physical object using a concise singular noun."
        )

        contents = [prompt_text]

        # 1. Full contextual frame: draw red bounding box on an in-memory copy (clean frame on disk untouched)
        first_crop_path, first_full_path, first_bbox = top_rows[0]
        if os.path.exists(first_full_path):
            clean_bgr = cv2.imread(first_full_path)
            if clean_bgr is not None:
                bx1, by1, bx2, by2 = first_bbox
                cv2.rectangle(clean_bgr, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                vis_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
                contents.append(Image.fromarray(vis_rgb))
            else:
                contents.append(Image.open(first_crop_path))
        else:
            contents.append(Image.open(first_crop_path))

        # 2. Add sharpest crop exemplars
        for crop_path, _, _ in top_rows:
            if os.path.exists(crop_path):
                contents.append(Image.open(crop_path))

        if self._client is None:
            if self.config.allow_mock_fallback:
                result = GeminiObjectSchema(
                    object_name="mock_discovered_object",
                    category="vehicle",
                    is_valid_physical_object=True,
                    confidence=0.88,
                    reasoning="Mock identifier fallback for unit tests and offline operation."
                )
            else:
                print("[Module 9] Warning: No GenAI client available. Returning None.")
                return None
        else:
            try:
                from google.genai import types

                response = self._client.models.generate_content(
                    model=self.config.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiObjectSchema,
                        temperature=self.config.temperature
                    )
                )

                result = GeminiObjectSchema.model_validate_json(response.text)
            except Exception as e:
                print(f"[Module 9] Error during Gemini VLM query for cluster {cluster_id}: {e}")
                if self.config.allow_mock_fallback:
                    result = GeminiObjectSchema(
                        object_name="mock_discovered_object",
                        category="vehicle",
                        is_valid_physical_object=True,
                        confidence=0.88,
                        reasoning=f"Mock identifier fallback due to API error: {e}"
                    )
                else:
                    return None

        new_status = "IDENTIFIED" if result.is_valid_physical_object else "NOISE"
        db.update_cluster_status(
            cluster_id=cluster_id,
            status=new_status,
            identified_label=result.object_name.strip().lower(),
            vlm_confidence=result.confidence,
            vlm_reasoning=result.reasoning
        )

        # Trigger Phase 3 Continual Retraining if valid physical object and confident
        if (new_status == "IDENTIFIED" and
                result.confidence >= self.config.min_confidence_for_retrain and
                retrain_queue is not None):
            try:
                retrain_queue.put(cluster_id)
            except Exception as q_err:
                print(f"[Module 9] Warning: Failed to enqueue cluster {cluster_id} to retrain_queue: {q_err}")

        return result
