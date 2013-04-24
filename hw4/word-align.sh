#!/bin/bash

# repeat the src sentences 100x
paste -d \\\n data/dev.src data/dev.src data/dev.src data/dev.src data/dev.src data/dev.src data/dev.src data/dev.src data/dev.src data/dev.src > data/dev.10xsrc
paste -d \\\n data/dev.10xsrc data/dev.10xsrc data/dev.10xsrc data/dev.10xsrc data/dev.10xsrc data/dev.10xsrc data/dev.10xsrc data/dev.10xsrc data/dev.10xsrc data/dev.10xsrc > data/dev.100xsrc 
paste -d \\\n data/test.src data/test.src data/test.src data/test.src data/test.src data/test.src data/test.src data/test.src data/test.src data/test.src > data/test.10xsrc
paste -d \\\n data/test.10xsrc data/test.10xsrc data/test.10xsrc data/test.10xsrc data/test.10xsrc data/test.10xsrc data/test.10xsrc data/test.10xsrc data/test.10xsrc data/test.10xsrc > data/test.100xsrc 

# remove non-words
python - <<END
def clean(input_filename, output_filename):
  with open(input_filename) as input_file, open(output_filename, mode='w') as output_file:
    for line in input_file:
        output_file.write(line.strip().split(' ||| ')[1]+'\n')
clean('data/dev.100best', 'data/dev.100best.tokens')
clean('data/dev.100xsrc', 'data/dev.100xsrc.tokens')
clean('data/test.100best', 'data/test.100best.tokens')
clean('data/test.100xsrc', 'data/test.100xsrc.tokens')
END

# paste
~/tools/cdec/corpus/paste-files.pl data/dev.100xsrc.tokens data/dev.100best.tokens > data/dev.100best.tokens.ru-en 
~/tools/cdec/corpus/paste-files.pl data/test.100xsrc.tokens data/test.100best.tokens > data/test.100best.tokens.ru-en 
cp data/dev.100best.tokens.ru-en data/align-this
cat data/test.100best.tokens.ru-en >> data/align-this
cat data/nc.lc.ru-en >> data/align-this

# fastalign
~/tools/cdec/word-aligner/fast_align -d -v -i data/align-this | head -n 120000 > data/align-this.align
head -n 40000 data/align-this.align > data/dev.100best.tokens.ru-en.align
tail -n 80000 data/align-this.align > data/test.100best.tokens.ru-en.align

# clean
rm data/dev.10xsrc data/dev.100xsrc data/dev.100xsrc.tokens data/dev.100best.tokens data/dev.100best.tokens.ru-en
rm data/test.10xsrc data/test.100xsrc data/test.100xsrc.tokens data/test.100best.tokens data/test.100best.tokens.ru-en
rm data/align-this data/align-this.align

