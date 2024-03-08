import streamlit as st
from io import BytesIO
import tensorflow as tf
from PIL import Image
import numpy as np
import re
from Deblurr_Autoencoder_Model import Deblurr_Autoencoder

class FileUpload(object):

    def __init__(self):
        self.fileTypes = [
            '.jpg',
            '.jpeg',
            '.png',
            '.gif',
            '.bmp',
            '.tiff',
            '.tif',
            '.webp',
            '.x-icon',
            '.tga',
            '.octet-stream'
        ]
        self.weights_loaded = False
        self.model_loaded = False

    def run(self):

        hide_label = self.custom_languages(lang="ID")

        # Inisialisasi flags untuk penanda apakah weights dan model sudah dimuat
        st.markdown(hide_label, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Unggah gambar Anda disini")
        col1, col2 = st.columns(2)

        if uploaded_file is not None:
            match = re.match(r'^.*?/(.*)$', uploaded_file.type)
            get_extension = "." + match.group(1)
            if get_extension not in self.fileTypes:
                st.error("File yang Anda unggah tidak termasuk dalam jenis file yang diizinkan.")
            else:
                col1.image(uploaded_file, caption='Gambar Blur')
                if col1.button('Rekonstruksi'):
                    recons_kernel = 200
                    low_resolution_shape = (recons_kernel, recons_kernel, 3)
                    input_low_resolution = tf.keras.Input(shape=low_resolution_shape)
                    autoencoder = Deblurr_Autoencoder(inputs=input_low_resolution)

                    autoencoder.load_weights('Deblurr_App/Deblurr model/Autoencoder model/checkpoint').expect_partial()

                    width_ori, height_ori, padded_img = self.pad_image(uploaded_file, recons_kernel)
                    cols, rows = tuple(map(lambda x: int(x / recons_kernel), padded_img.size))
                    img_rekons = self.split_images(padded_img, cols, rows, autoencoder)
                    img_rekons = self.combine_images(img_rekons, cols, rows)
                    img_rekons = Image.fromarray(img_rekons)
                    pil_img = img_rekons.crop((0, 0, width_ori, height_ori))

                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    col2.image(pil_img, caption='Gambar rekonstruksi')

                    # Tambahkan tombol unduh
                    col2.download_button(
                        label="Unduh Gambar",
                        data=byte_im,
                        file_name="reconstructed_image.png",
                        mime="image/png",
                    )
    def pad_image(self, image_path, recons_kernel=150):
        """Pad an image with a specified reconstruction kernel size."""

        image = Image.open(image_path)
        img_array = np.array(image.convert('RGB'))
        max_values = np.max(img_array, axis=(0, 1))
        padding_color = tuple(map(int, max_values))

        original_width, original_height = image.size

        right_pad = recons_kernel - (original_width % recons_kernel)
        bottom_pad = recons_kernel - (original_height % recons_kernel)

        new_width, new_height = original_width + right_pad, original_height + bottom_pad
        padded_image = Image.new(mode="RGB", size=(new_width, new_height), color=padding_color)
        padded_image.paste(image)

        return original_width, original_height, padded_image

    def recons_single_patch(self, patch, autoencoder):
        """Process a single image patch using the given models."""
        image = np.array(patch) / 255.0
        image = tf.maximum(image, 0)
        image = tf.expand_dims(image, axis=0)

        img_deblurr = np.clip(autoencoder.predict(image, verbose=0), 0.0, 1.0)
        img_deblurr = img_deblurr * 255
        img_deblurr = img_deblurr.astype(np.uint8)

        return img_deblurr[0]

    def split_images(self, padded_img, cols, rows, autoencoder):
        """Process images in a grid using the given models."""
        my_image = []

        patch_width, patch_height = padded_img.size[0] // cols, padded_img.size[1] // rows

        for i in range(cols):
            for j in range(rows):
                left, upper = i * patch_width, j * patch_height
                right, lower = left + patch_width, upper + patch_height

                cropped_im = padded_img.crop((left, upper, right, lower))
                img_pred = self.recons_single_patch(cropped_im, autoencoder)
                my_image.append(img_pred)

        return my_image

    def combine_images(self, image_list, cols, rows):
        """Combine a list of images into a single image grid."""
        width, height = image_list[0].shape[1], image_list[0].shape[0]
        result = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)

        for i in range(cols):
            for j in range(rows):
                left, upper = i * width, j * height
                right, lower = left + width, upper + height
                result[upper:lower, left:right, :] = image_list[i * rows + j]

        return result

    def custom_languages(self, lang="ID"):
        languages = {
            "EN": {
                "button": "Browse Files",
                "instructions": "Drag and drop files here",
                "limits": "Limit 200MB per file",
            },

            "ID": {
                "button": "Telusuri File",
                "instructions": "Seret dan letakkan file disini",
                "limits": "Batas 200MB per file",
            },
        }

        hide_label = (
            """ 
            <style>
                div [data-testid="stFileDropzoneInstructions"]>div>small,
                div [data-testid="stFileDropzoneInstructions"]>div>span,
                div [data-testid="stFileUploader"]>section>button
                {
                    visibility:hidden;
                    position:relative;
                    
                }
                
                div [data-testid="stFileUploader"]>section>button::after
                {
                    content: "BUTTON_TEXT";
                    visibility:visible;
                    position:absolute;
                    width:fit-content;
                    border-radius:.5rem;
                    padding:.4rem .70rem;
                    background:#0e1117;
                    border:1px solid #acb1c3;
                    transition:.25s ease;
                    width:max-content;
                }

                div [data-testid="stFileUploader"]>section>button:hover::after
                {
                    border:1px solid red;
                }

                div [data-testid="stFileDropzoneInstructions"]>div>span::after,
                div [data-testid="stFileDropzoneInstructions"]>div>small::after 
                {
                    visibility:visible;
                    position:absolute;
                    left:0;
                }

                div [data-testid="stFileDropzoneInstructions"]>div>span::after 
                {
                    content:"INSTRUCTIONS_TEXT";
                    width:max-content;
                    font-size:14px;
                }

                div [data-testid="stFileDropzoneInstructions"]>div>small::after 
                {
                    content:"FILE_LIMITS";
                    font-size:14px;
                }
          </style>
 
          """
            .replace("BUTTON_TEXT", languages.get(lang).get("button"))
            .replace("INSTRUCTIONS_TEXT", languages.get(lang).get("instructions"))
            .replace("FILE_LIMITS", languages.get(lang).get("limits"))
        )

        return hide_label

if __name__ == "__main__":
    helper = FileUpload()
    helper.run()