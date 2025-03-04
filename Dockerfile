# Start with the official CUDA image (which typically includes the latest stable version)
FROM  python:3.12

RUN pip install torch tqdm --index-url https://download.pytorch.org/whl/cu126
RUN pip install transformers nltk

ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} fast && \ 
    useradd -m -u ${UID} -g ${GID} -s /bin/bash fast


COPY . /home/fast/FAST
RUN chown -R fast:fast /home/fast/FAST
# RUN mkdir -p ~/FAST/data &&  wget https://storage.googleapis.com/grover-models/generation_examples/generator=mega~dataset=p0.94.jsonl -O ~/FAST/data/p0.94.jsonl

WORKDIR /home/fast/FAST
USER fast

CMD ["/bin/bash"]