import streamlit as st
import requests
from PIL import Image
from bird.pretrained_model import Model
from torchvision.transforms import ToTensor
import torch

def main():
    st.title("Indian Bird Classification")
    st.write("Upload an image and click the button to classify the bird.")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, use_column_width=True)

        if st.button("Classify"):
            # Resize the image
            img = Image.open(uploaded_file)
            img = img.resize((224,224))
            img = ToTensor()(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to("cuda")

            model = Model()
            model.to("cuda")
            model.model.load_state_dict(torch.load(r'D:\Coding\bird_detection\Bird_Detection\models\968857.pth'))
            label,confidence = model.prediction(img)
            st.write(label)
            st.write(confidence)
            
if __name__ == "__main__":
    main()