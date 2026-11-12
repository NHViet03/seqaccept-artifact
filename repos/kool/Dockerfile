# https://github.com/edasi/kool at the IDoFT commit. Build: docker build -t seqaccept/kool:1 repos/kool
FROM seqaccept/base:1
RUN git clone -q https://github.com/edasi/kool /repo && git -C /repo checkout -q efe847a05c1a07af5eddeeeb700fdefcb6c9ced9
WORKDIR /repo
RUN pip install --no-cache-dir  -e .
