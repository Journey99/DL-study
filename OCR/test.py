import easyocr

reader = easyocr.Reader(['ko', 'en'])
result = reader.readtext('test.jpg')

for bbox, text, score in result:
    print(f"{text} (conf: {score:.4f})")

sorted_result = sorted(result, key=lambda x: x[2], reverse=True)
for bbox, text, score in sorted_result:
    print(f"{text} (conf: {score:.4f})")

'''
([[np.int32(95), np.int32(99)], [np.int32(136), np.int32(99)], [np.int32(136), np.int32(242)], [np.int32(95), np.int32(242)]], '#', np.float64(0.07458661529291533))
: 앞에 4개는 글자 영역 / 인식한 텍스트 / confidence
'''

