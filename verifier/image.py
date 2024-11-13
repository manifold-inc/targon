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

def generate_image_functions(MODEL_WRAPPER,MODEL_NAME,ENDPOINTS):
    async def generate_ground_truth(prompt,width, height):
        image = MODEL_WRAPPER(prompt, height=int(height), width=int(width)).images[0]  # type: ignore
        buffered = BytesIO()
        image.save(buffered, format="png")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str

    async def verify():

        pass
    return verify

