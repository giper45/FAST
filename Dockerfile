# Start with the official CUDA image (which typically includes the latest stable version)
FROM  python:3.8
# The largest task when high data are present

RUN pip install torch tqdm --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers==2.9.0 nltk pathos fuzzywuzzy python-Levenshtein scikit-learn tensorboardx six==1.17.0 wikipedia2vec pandas datasets==2.10.1

ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} fast && \ 
    useradd -m -u ${UID} -g ${GID} -s /bin/bash fast

# RUN wget https://huggingface.co/jinmang2/dooly-hub/resolve/0fcf5b24ba3748253300579ecaac0de546aec668/word_embedding/en/wikipedia2vec.en/enwiki_20180420_100d.pkl -O /home/fast/FAST/data/enwiki_20180420_100d.pkl
COPY . /home/fast/FAST
RUN chown -R fast:fast /home/fast/FAST
# RUN mkdir -p ~/FAST/data &&  wget https://storage.googleapis.com/grover-models/generation_examples/generator=mega~dataset=p0.94.jsonl -O ~/FAST/data/p0.94.jsonl

WORKDIR /home/fast/FAST
USER fast

CMD ["/bin/bash"]