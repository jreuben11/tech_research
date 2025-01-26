import re

def extract(input_file, output_file, topic):
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        capture = False
        for line in lines:
            if topic in line:
                capture = True
            if capture:
                outfile.write("- " + line)
                if not re.match(r'^\s*[-*0-9]', line):
                    capture = False

input_file = 'old_links.md'
output_file = 'paperswithcode_links.md'
topic = 'paperswithcode'

extract(input_file, output_file, topic)