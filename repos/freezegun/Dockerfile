# https://github.com/spulec/freezegun at the IDoFT commit. Build: docker build -t seqaccept/freezegun:1 repos/freezegun
FROM seqaccept/base:1
RUN git clone -q https://github.com/spulec/freezegun /repo && git -C /repo checkout -q b46da782a7a051081fd51577749cfc0074db0cc6
WORKDIR /repo
RUN pip install --no-cache-dir  -e .
