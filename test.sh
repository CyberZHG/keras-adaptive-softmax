#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_adaptive_softmax tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_adaptive_softmax --with-doctest