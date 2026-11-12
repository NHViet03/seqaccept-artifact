# https://github.com/didix21/mdutils at the IDoFT commit. Build: docker build -t seqaccept/mdutils:1 repos/mdutils
FROM seqaccept/base:1
RUN git clone -q https://github.com/didix21/mdutils /repo && git -C /repo checkout -q 2ef859ecc1e4f1a3e24f4150c96f61180bc57ceb
WORKDIR /repo
RUN pip install --no-cache-dir  -e .
