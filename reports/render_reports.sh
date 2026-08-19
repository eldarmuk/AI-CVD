#!/bin/bash

cd "$(dirname "$0")"

echo "Rendering PDF..."
quarto render ../notebooks/02_comprehensive_eda.ipynb --to pdf

echo "Rendering HTML..."
quarto render ../notebooks/02_comprehensive_eda.ipynb --to html

echo "Moving rendered files to the current folder..."
mv -f ../notebooks/02_comprehensive_eda.pdf ./02_comprehensive_eda.pdf
mv -f ../notebooks/02_comprehensive_eda.html ./02_comprehensive_eda.html

echo "Success! Files are in the reports folder."
