#Code for paper SeburSum: A Novel Set-based Summary Ranking Strategy for Summary-level Extractive Summarization#

We publish the set-based summary ranking strategy (SeburSum) code, best checkpoints and test data to evaluate the results.
Contrastive learning training code will be released soon！

The test data is available at :<>

The  checkpoints are available <>



After summaries are extracted, you should use ROUGE to score each summary.  In order to get correct ROUGE scores, we recommend using the following commands to install the ROUGE environment:

```bash
sudo apt-get install libxml-perl libxml-dom-perl
git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
python setup.py install
export PYROUGE_HOME_DIR=the/path/to/RELEASE-1.5.5
pyrouge_set_rouge_path $PYROUGE_HOME_DIR
chmod +x $PYROUGE_HOME_DIR/ROUGE-1.5.5.pl
```

You can refer to https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5 for RELEASE-1.5.5 and remember to build  Wordnet 2.0 instead of 1.6 in RELEASE-1.5.5/data:

```
cd $PYROUGE_HOME_DIR/data/WordNet-2.0-Exceptions/
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
rm WordNet-2.0.exc.db
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

