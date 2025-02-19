
source venv/bin/activate
python3 ice.py	                  	    \
    -k 10                       	    \
    -n 10                       	    \
    --pval-consider full-train  	    \
    -t quartiles                	    \
    --q-consider correct              \
    -c cred
