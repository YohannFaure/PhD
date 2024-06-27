#!/bin/bash

# tell it to stop on any error
handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

cd ./These_latex

echo "Checking if removing ./build is needed"

remove_build="false"
# Check differences between preamble.sty and ./build/preamble.sty
if diff -q preamble.sty ./build/preamble.sty >/dev/null 2>&1; then
    echo "No difference found between preamble.sty and ./build/preamble.sty"
else
    echo "Difference found between preamble.sty and ./build/preamble.sty"
    remove_build="true"
fi

# Check differences between thesebib.bib and ./build/thesebib.bib
if diff -q "thesebib.bib" "./build/thesebib.bib" >/dev/null 2>&1; then
    echo "No difference found between thesebib.bib and ./build/thesebib.bib"
else
    echo "Difference found between thesebib.bib and ./build/thesebib.bib"
    remove_build="true"
fi

# removing
if $remove_build == "true"; then
    echo "Removing ./build"
    rm -rf "./build"
fi


# copy files in ./build
echo ""
echo "Copying files in the build directory"

rsync -ar ./ ./build --exclude build --exclude compile.sh > /dev/null
cp -r ../Figures_chap* ./
cp -r ../logos ./

cd ./build/
rm -rf *.bak

# Running the compilation in itself
echo ""
echo "pdfLaTeX 1"
taskset -c 1 pdflatex -synctex=1 -interaction=nonstopmode --shell-escape these.tex > compil_log.log
echo "BibTeX"
bibtex these.aux  > compil_bibtex_log.log
echo "pdfLaTeX 2"
taskset -c 3 pdflatex -synctex=1 -interaction=nonstopmode --shell-escape these.tex > compil_log.log
echo "pdfLaTeX 3"
taskset -c 5 pdflatex -synctex=1 -interaction=nonstopmode --shell-escape these.tex > compil_log.log


# cleaning and saving
echo ""
echo "Cleaning and saving files"
# removing junk in the ./build directory
rm *.tex

# copy pdf file
cp these.pdf ../

cd ../
rm -rf ./Figures_chap*
rm -rf ./logos

# open file
#xdg-open these.pdf


