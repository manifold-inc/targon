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


## TODO build verification for images
def generate_image_functions(MODEL_WRAPPER,MODEL_NAME,ENDPOINTS):

    ## cache this across requests for the same inputs
    async def generate_ground_truth(prompt,width, height):
        image = MODEL_WRAPPER(prompt, height=int(height), width=int(width)).images[0]  # type: ignore
        buffered = BytesIO()
        image.save(buffered, format="png")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str

    async def verify(ground_truth, miner_response):
        pass

    return verify

