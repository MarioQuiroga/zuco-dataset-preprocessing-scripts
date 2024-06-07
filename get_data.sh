#!/bin/bash

#Install osf-client
echo "Installing osf-client"
pip install osfclient

echo "Creating data dir"
mkdir data

echo "Downloading zuco-dataset from osf, this may take a while"
osf -p 2urht clone
echo "Finished downloading zuco-dataset now cleaning"

mv 2urht/dropbox data/
rm -r 2urht