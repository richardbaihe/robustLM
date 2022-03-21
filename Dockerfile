FROM nvcr.io/nvidia/pytorch:21.02-py3
WORKDIR /root
# -------pip packages installing-------
RUN pip install google-cloud-storage
# RUN echo "Installing Apex on top of ${BASE_IMAGE}"
# # uninstall Apex if present, twice to make absolutely sure :)
# RUN pip uninstall -y apex || :
# RUN pip uninstall -y apex || :

# RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
# WORKDIR /root/apex
# RUN git checkout bd6e66df95840c92e6dff3a8f38149bb3c5dbcd6
# COPY ./setup.py /workspace/apex/setup.py
# RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# WORKDIR /root

RUN pip install nltk
RUN python -m nltk.downloader punkt wordnet
RUN pip install wandb
RUN pip install tensorboard
RUN pip install tokenizers
RUN pip install mosestokenizer

# Download the data from the public Google Cloud Storage bucket for this sample
COPY ./data/wikitext-103 ./robustLM/data/wikitext-103
COPY ./data/arxiv/timed ./robustLM/data/arxiv/timed

# Copies the trainer code to the docker image.
COPY ./src ./robustLM/src
WORKDIR /root/robustLM/src
# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-u", "train.py"]       