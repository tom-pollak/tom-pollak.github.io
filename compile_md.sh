echo "*****"
mkdir -p pages

# Process both markdown and Jupyter notebook files
for file in md/*.md nbs/*.ipynb; do
  filename=$(basename "$file" | cut -f 1 -d '.')
  pandoc "$file" -o "pages/$filename.html" \
    --template=markdown_template.html \
    --metadata title="$filename" \
    --mathjax
  echo "processed $filename"
done

echo "*****"
