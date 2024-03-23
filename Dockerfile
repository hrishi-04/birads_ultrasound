FROM python:3.11.4
WORKDIR /app
# This copies only the selected files in your current directory to the /app directory in the container.
COPY ultrasound_birad_model.hdf5 /app
COPY requirements.txt /app
COPY app.py /app
COPY info.csv /app
COPY test_inst /app/test_inst



RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# This tells Docker to listen on port 80 at runtime. Port 80 is the standard port for HTTP.
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

# This command tells Streamlit to run your app.py script when the container starts.
CMD ["app.py"]