from PyPDF2 import PdfFileMerger

def file_merger(files,output):
	pfm = PdfFileMerger()
	for file in files:
		pfm.append(file)
	pfm.write(output)
	pfm.close()
