# https://github.com/hefnawi/json-storage-manager at the IDoFT commit. Build: docker build -t seqaccept/jsm:1 repos/jsm
FROM seqaccept/base:1
RUN git clone -q https://github.com/hefnawi/json-storage-manager /repo && git -C /repo checkout -q c7521fc4a576cf23a8c2454106bed6fb8c951b8d
WORKDIR /repo
RUN pip install --no-cache-dir six -e .
