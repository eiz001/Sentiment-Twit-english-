from textblob import TextBlob
import pandas as pd
import re
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)


# In[23]:


Filecsv = input('Path CSV file: ')
Kolom= input ('Masukan Nama Kolom:')
Titlechart = input('Judul Chart: ')
data = pd.read_csv (Filecsv)
df = pd.DataFrame(data, columns= [Kolom])
df.head()


# In[24]:


def cleanUpText(txt):
    txt = re.sub(r'@[A-Za-z0-9_]+','',txt)
    txt = re.sub(r'#','',txt)
    txt = re.sub(r'RT : ','',txt)
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',txt)
    return txt


# In[25]:


df['Text'] = df['Text'].apply(cleanUpText)


# In[26]:


def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity


# In[27]:


def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity


# In[28]:


df['Subjectivity'] = df ['Text'].apply(getTextSubjectivity)
df['Polarity'] = df ['Text'].apply(getTextPolarity)


# In[29]:


df.head(50)


# In[30]:


df = df.drop (df[df['Text']==''].index)


# In[40]:


df.head (50)


# In[41]:


def getTextAnalysis(a):
    if a<0:
        return "Negative"
    elif a==0:
        return"Neutral"
    else:
        return"Positive"


# In[42]:


df['Score'] = df ['Polarity'].apply(getTextAnalysis)


# In[34]:


df.head (50)


# In[35]:


positive = df[df['Score']=="Positive"]
print(str(positive.shape[0]/(df.shape[0])*100) + "% of positive tweets")
pos = positive.shape[0]/df.shape[0]*100


# In[36]:


negative = df [df['Score']=="Negative"]
print(str(negative.shape[0]/(df.shape[0])*100) + "% of Negative tweets")
neg=negative.shape[0]/df.shape[0]*100


# In[37]:


neutral = df [df['Score']=="Neutral"]
print(str(neutral.shape[0]/(df.shape[0])*100) + "% of Neutaral tweets")
neutrall=neutral.shape[0]/df.shape[0]*100


# In[38]:


explode=(0,0.1,0)
labels = 'Positive','Negative','Neutral'
sizes = [pos,neg,neutrall]
colors = ['red','purple','gold']


# In[39]:


plt.pie(sizes,explode=explode,colors=colors,autopct= '%1.1f%%',startangle=120)
plt.legend(labels,loc=(-0.05,0.05),shadow=True)
plt.axis('equal')
plt.title(Titlechart)
plt.savefig("Sentimen.png")
plt.show()
