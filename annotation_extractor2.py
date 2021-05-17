import sys
import PyPDF2, traceback

try :
    src = sys.argv[1]
except :
    src = r'/home/gray/wk/papers/first-resubmissions/ThomasEA2021TRO/PdfComments7.pdf'


pdf_object = PyPDF2.PdfFileReader(open(src, "rb"))

nPages = pdf_object.getNumPages()
n=0

for i in range(nPages) :
    page = pdf_object.getPage(i)
    # try :
    if '/Annots' in page:
        for annot in page['/Annots'] :
            # print(dir(annot))
            # print(dir(annot.getObject()))
            # print(annot.getObject().keys())
            # exit()
            # print (annot.getObject())       # (1)
            n+=1
            print(n, annot.getObject()["/Subtype"])
            if '/Popup' in annot.getObject():
                poptype = (type(annot.getObject()["/Popup"]["/Contents"]))
                if poptype==PyPDF2.generic.ByteStringObject:
                    print((annot.getObject()["/Popup"]["/Contents"]).decode("utf-16"))
                elif poptype==PyPDF2.generic.TextStringObject:
                    print(annot.getObject()["/Popup"]["/Contents"])
                else: print(poptype)
            elif "/Contents" in annot.getObject():
                content = annot.getObject()['/Contents']
                if type(content)==PyPDF2.generic.ByteStringObject:
                    print(content.decode("utf-16"))
                elif type(content)==PyPDF2.generic.TextStringObject:
                    print(content)
                else:
                    print(type(content))
            else:
                # print("nope")
                pass

            if str(annot.getObject()["/Subtype"])=="/Highlight":
                pass
                # print(type(str(annot.getObject()["/Subtype"])))
                # print(str(annot.getObject()["/Subtype"])=="/Highlight")
                # print(annot.getObject().keys())

            # print ('')
    # except : 
        # there are no annotations on this page
        # pass