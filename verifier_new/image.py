####################################
#   ___                            
#  |_ _|_ __ ___   __ _  __ _  ___ 
#   | || '_ ` _ \ / _` |/ _` |/ _ \
#   | || | | | | | (_| | (_| |  __/
#  |___|_| |_| |_|\__,_|\__, |\___|
#                       |___/      
#
####################################


import base64
from io import BytesIO
import logging


## TODO build verification for images
def generate_image_functions(MODEL_WRAPPER, MODEL_NAME ,ENDPOINTS, xgb_model):

    ## cache this across requests for the same inputs
    async def generate_ground_truth(prompt,width, height):
        image = MODEL_WRAPPER(prompt, height=int(height), width=int(width)).images[0]  # type: ignore
        buffered = BytesIO()
        image.save(buffered, format="png")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str

    async def verify(ground_truth, miner_response):
        try:
            # Verify internal consistency of miner's response
            clip_embedding_miner = await _query_endpoint_clip_embeddings({"image_b64s": [miner_response]})
            clip_embedding_miner = clip_embedding_miner.clip_embeddings[0]
            
            # Get CLIP embeddings for ground truth
            clip_embedding_truth = await _query_endpoint_clip_embeddings({"image_b64s": [ground_truth]})
            clip_embedding_truth = clip_embedding_truth.clip_embeddings[0]
            
            # Calculate image hashes
            miner_hashes = checking_utils.calculate_image_hashes(miner_response)
            truth_hashes = checking_utils.calculate_image_hashes(ground_truth)
            
            # Compare hashes
            hash_distances = checking_utils.get_hash_distances(miner_hashes, truth_hashes)
            
            # Get similarity prediction from XGBoost model
            probability_same_image = xgb_model.predict_proba([hash_distances])[0][1]
            
            # Calculate CLIP embedding similarity
            clip_similarity = checking_utils.get_clip_embedding_similarity(
                clip_embedding_miner, 
                clip_embedding_truth
            )
            
            # Calculate final score using weighted combination
            score = float(probability_same_image**0.5) * 0.4 + (clip_similarity**2) * 0.6
            
            return 1 if score > 0.95 else score**2
            
        except Exception as e:
            logger.error(f"Error in image verification: {str(e)}")
            return 0

    return verify

