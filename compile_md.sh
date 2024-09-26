mkdir -p pages
for file in md/*.md; do
  filename=$(basename "$file" .md)
  pandoc "$file" -o "pages/$filename.html" \
    --template=markdown_template.html \
    --mathjax
done
