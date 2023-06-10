from transformers import  MBartForConditionalGeneration, MBart50TokenizerFast
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

m = AutoModelForSeq2SeqLM.from_pretrained("../mt5_testrun/checkpoint-400")

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")


article_hi = ["प्रधानमंत्री ने कहा कि बाबा साहेब अम्बेडकर की करोड़ों लोगों के दिलों एवं दिमाग में 'आकांक्षा' पैदा करने में मुख्य भूमिका थी ।",
"प्रधानमंत्री ने कहा कि सरकार स्पष्ट लक्ष्यों और समयसीमा के साथ विभिन्न योजनाओं पर कार्य कर रही है ।",
"उन्होंने कहा कि संरचना विकसित करने की गति काफी बढ़ी है ।",
"अपने संदेश में प्रधानमंत्री ने कहा  राज्य के स्थापना दिवस पर नगालैंड वासियों को बधाई ।",
"उन्होंने कहा कि हर पीढ़ी संविधान से अच्छी तरह परिचित होनी चाहिए और इसे समकालीन संदर्भ में याद किया जाना चाहिए ।"]


inputs = tokenizer(article_hi, return_tensors="pt",max_length=210,truncation=True,padding='max_length')


output = m.generate(**inputs,num_beams=10)
print(tokenizer.batch_decode(output, skip_special_tokens=True))