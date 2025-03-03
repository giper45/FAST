FROM python:3.8

ARG UID=1000
ARG GID=1000

RUN pip install nltk allennlp==1.0.0 allennlp-models==1.0.0 tqdm==4.67.1 spacy==2.2.4
RUN python -m spacy download en_core_web_sm


RUN groupadd -g ${GID} fast && \ 
    useradd -m -u ${UID} -g ${GID} -s /bin/bash fast


COPY . /home/fast/FAST
RUN chown -R fast:fast /home/fast/FAST
# RUN mkdir -p ~/FAST/data &&  wget https://storage.googleapis.com/grover-models/generation_examples/generator=mega~dataset=p0.94.jsonl -O ~/FAST/data/p0.94.jsonl

WORKDIR /home/fast/FAST
USER fast



CMD ["/bin/bash", "/home/fast/FAST/docker_entrypoint.sh"]