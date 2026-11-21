import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import export_docx as ed
ed.SRC = os.path.join(ed.ROOT, 'paper', 'PAPER_EN_final.md')
ed.OUT = os.path.join(ed.ROOT, 'paper', 'HCM_PAPER_EN.docx')
sys.exit(ed.main())