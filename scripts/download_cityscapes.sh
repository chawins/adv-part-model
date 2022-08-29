#!/bin/bash
wget --keep-session-cookies --save-cookies=cookies.txt \
     --post-data "username=$1&password=$2&submit=Login" https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt \
     --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
wget --load-cookies cookies.txt \
     --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=35
