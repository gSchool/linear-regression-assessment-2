FROM python:3.8

# Makes directory and changes working directory to it
WORKDIR /app

# Installs packages needed for running test.py
COPY test_requirements.txt ./
RUN pip install --no-cache-dir -r ./test_requirements.txt

# Student submission from Learn
ARG SUBMISSION_SUBFOLDER
COPY $SUBMISSION_SUBFOLDER ./
# Anything hereafter is rebuilt every time the Dockerfile is run
COPY solutions.py ./$SUBMISSION_SUBFOLDER

# Bring fresh copies of test.sh and test.py into Docker image (in case student modified these on their fork)
COPY test.py ./
COPY test.sh ./

CMD [ "python", "./test.py"]

