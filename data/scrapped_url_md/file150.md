- [Python Tutorial](https://www.geeksforgeeks.org/python-programming-language-tutorial/)
- [Interview Questions](https://www.geeksforgeeks.org/python-interview-questions/)
- [Python Quiz](https://www.geeksforgeeks.org/python-quizzes/)
- [Python Glossary](https://www.geeksforgeeks.org/python-glossary/)
- [Python Projects](https://www.geeksforgeeks.org/python-projects-beginner-to-advanced/)
- [Practice Python](https://www.geeksforgeeks.org/python-exercises-practice-questions-and-solutions/)
- [Data Science With Python](https://www.geeksforgeeks.org/data-science-with-python-tutorial/)
- [Python Web Dev](https://www.geeksforgeeks.org/python-web-development-django/)
- [DSA with Python](https://www.geeksforgeeks.org/python-data-structures-and-algorithms/)
- [Python OOPs](https://www.geeksforgeeks.org/python-oops-concepts/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/normalizing-textual-data-with-python/?type%3Darticle%26id%3D541250&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
How to change default font in Tkinter?\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/how-to-change-default-font-in-tkinter/)

# Normalizing Textual Data with Python

Last Updated : 26 Nov, 2022

Comments

Improve

Suggest changes

2 Likes

Like

Report

In this article, we will learn How to Normalizing Textual Data with Python. Let’s discuss some concepts :

- **Textual data** ask systematically collected material consisting of written, printed, or electronically published words, typically either purposefully written or transcribed from speech.
- **Text normalization** is that the method of transforming text into one canonical form that it’d not have had before. Normalizing text before storing or processing it allows for separation of concerns since the input is sure to be consistent before operations are performed thereon. Text normalization requires being conscious of what sort of text is to be normalized and the way it’s to be processed afterwards; there’s no all-purpose normalization procedure.

**Steps Required**

Here, we will discuss some basic steps need for Text normalization.

- Input text String,
- Convert all letters of the string to one case(either lower or upper case),
- If numbers are essential to convert to words else remove all numbers,
- Remove punctuations, other formalities of grammar,
- Remove white spaces,
- Remove stop words,
- And any other computations.

We are doing Text normalization with above-mentioned steps, every step can be done in some ways. So we will discuss each and everything in this whole process.

**Text String**

- Python3

## Python3

|     |
| --- |
| `# input string `<br>`string ` `=` `"       Python 3.0, released in 2008, was a major revision of the language that is not completely backward compatible and much Python 2 code does not run unmodified on Python 3. With Python 2's end-of-life, only Python 3.6.x[30] and later are supported, with older versions still supporting e.g. Windows 7 (and old installers not restricted to 64-bit Windows)."`<br>`print` `(string)` |

```

```

```

```

**Output:**

> ”       Python 3.0, released in 2008, was a major revision of the language that is not completely backward compatible and much Python 2 code does not run unmodified on Python 3. With Python 2’s end-of-life, only Python 3.6.x\[30\] and later are supported, with older versions still supporting e.g. Windows 7 (and old installers not restricted to 64-bit Windows).”

**Case Conversion (** [**Lower Case**](https://www.geeksforgeeks.org/isupper-islower-lower-upper-python-applications/#:~:text=In%20Python%2C%20lower()%20is,all%20uppercase%20characters%20to%20lowercase.) **)**

In Python, lower() is a built-in method used for string handling. The lower() methods returns the lowercased string from the given string. It converts all uppercase characters to lowercase. If no uppercase characters exist, it returns the original string.

- Python3

## Python3

|     |
| --- |
| `# input string`<br>`string ` `=` `"       Python 3.0, released in 2008, was a major revision of the language that is not completely backward compatible and much Python 2 code does not run unmodified on Python 3. With Python 2's end-of-life, only Python 3.6.x[30] and later are supported, with older versions still supporting e.g. Windows 7 (and old installers not restricted to 64-bit Windows)."`<br>`# convert to lower case`<br>`lower_string ` `=` `string.lower()`<br>`print` `(lower_string)` |

```

```

```

```

**Output:**

> ”       python 3.0, released in 2008, was a major revision of the language that is not completely backward compatible and much python 2 code does not run unmodified on python 3. with python 2’s end-of-life, only python 3.6.x\[30\] and later are supported, with older versions still supporting e.g. windows 7 (and old installers not restricted to 64-bit windows).”

[**Removing Numbers**](https://www.geeksforgeeks.org/python-remove-all-digits-from-a-list-of-strings/)

Remove numbers if they’re not relevant to your analyses. Usually, regular expressions are used to remove numbers.

- Python3

## Python3

|     |
| --- |
| `# import regex`<br>`import` `re`<br>`# input string `<br>`string ` `=` `"       Python 3.0, released in 2008, was a major revision of the language that is not completely backward compatible and much Python 2 code does not run unmodified on Python 3. With Python 2's end-of-life, only Python 3.6.x[30] and later are supported, with older versions still supporting e.g. Windows 7 (and old installers not restricted to 64-bit Windows)."`<br>`# convert to lower case`<br>`lower_string ` `=` `string.lower()`<br>`# remove numbers`<br>`no_number_string ` `=` `re.sub(r` `'\d+'` `,'',lower_string)`<br>`print` `(no_number_string)` |

```

```

```

```

**Output:**

> ”       python ., released in , was a major revision of the language that is not completely backward compatible and much python  code does not run unmodified on python . with python ‘s end-of-life, only python ..x\[\] and later are supported, with older versions still supporting e.g. windows  (and old installers not restricted to -bit windows).”

[**Removing punctuation**](https://www.geeksforgeeks.org/python-remove-punctuation-from-string/)

The part of replacing with punctuation can also be performed using regex. In this, we replace all punctuation by empty string using certain regex.

- Python3

## Python3

|     |
| --- |
| `# import regex`<br>`import` `re`<br>`# input string `<br>`string ` `=` `"       Python 3.0, released in 2008, was a major revision of the language that is not completely backward compatible and much Python 2 code does not run unmodified on Python 3. With Python 2's end-of-life, only Python 3.6.x[30] and later are supported, with older versions still supporting e.g. Windows 7 (and old installers not restricted to 64-bit Windows)."`<br>`# convert to lower case`<br>`lower_string ` `=` `string.lower()`<br>`# remove numbers`<br>`no_number_string ` `=` `re.sub(r` `'\d+'` `,'',lower_string)`<br>`# remove all punctuation except words and space`<br>`no_punc_string ` `=` `re.sub(r` `'[^\w\s]'` `,'', no_number_string) `<br>`print` `(no_punc_string)` |

```

```

```

```

**Output:**

> ‘       python  released in  was a major revision of the language that is not completely backward compatible and much python  code does not run unmodified on python  with python s endoflife only python x and later are supported with older versions still supporting eg windows  and old installers not restricted to bit windows’

[**Removing White space**](https://www.geeksforgeeks.org/python-string-strip/)

The **strip()** function is an inbuilt function in Python programming language that returns a copy of the string with both leading and trailing characters removed (based on the string argument passed).

- Python3

## Python3

|     |
| --- |
| `# import regex`<br>`import` `re`<br>`# input string `<br>`string ` `=` `"       Python 3.0, released in 2008, was a major revision of the language that is not completely backward compatible and much Python 2 code does not run unmodified on Python 3. With Python 2's end-of-life, only Python 3.6.x[30] and later are supported, with older versions still supporting e.g. Windows 7 (and old installers not restricted to 64-bit Windows)."`<br>`# convert to lower case`<br>`lower_string ` `=` `string.lower()`<br>`# remove numbers`<br>`no_number_string ` `=` `re.sub(r` `'\d+'` `,'',lower_string)`<br>`# remove all punctuation except words and space`<br>`no_punc_string ` `=` `re.sub(r` `'[^\w\s]'` `,'', no_number_string) `<br>`# remove white spaces`<br>`no_wspace_string ` `=` `no_punc_string.strip()`<br>`print` `(no_wspace_string)` |

```

```

```

```

**Output:**

> ‘python  released in  was a major revision of the language that is not completely backward compatible and much python  code does not run unmodified on python  with python s endoflife only python x and later are supported with older versions still supporting eg windows  and old installers not restricted to bit windows’

**Removing Stop Words**

Stop words” are the foremost common words during a language like “the”, “a”, “on”, “is”, “all”. These words don’t carry important meaning and are usually faraway from texts. It is possible to get rid of stop words using tongue Toolkit (NLTK), a set of libraries and programs for symbolic and statistical tongue processing.

- Python3

## Python3

|     |
| --- |
| `# download stopwords`<br>`import` `nltk`<br>`nltk.download(` `'stopwords'` `)`<br>`# import nltk for stopwords`<br>`from` `nltk.corpus ` `import` `stopwords`<br>`stop_words ` `=` `set` `(stopwords.words(` `'english'` `))`<br>`print` `(stop_words)`<br>`# assign string`<br>`no_wspace_string` `=` `'python  released in  was a major revision of the language that is not completely backward compatible and much python  code does not run unmodified on python  with python s endoflife only python x and later are supported with older versions still supporting eg windows  and old installers not restricted to bit windows'`<br>`# convert string to list of words`<br>`lst_string ` `=` `[no_wspace_string][` `0` `].split()`<br>`print` `(lst_string)`<br>`# remove stopwords`<br>`no_stpwords_string` `=` `""`<br>`for` `i ` `in` `lst_string:`<br>`    ``if` `not` `i ` `in` `stop_words:`<br>`        ``no_stpwords_string ` `+` `=` `i` `+` `' '`<br>`        `<br>`# removing last space`<br>`no_stpwords_string ` `=` `no_stpwords_string[:` `-` `1` `]`<br>`print` `(no_stpwords_string)` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20210113223759/Screenshot345.png)

In this, we can normalize the textual data using Python. Below is the complete python program:

- Python3

## Python3

|     |
| --- |
| `# import regex`<br>`import` `re`<br>`# download stopwords`<br>`import` `nltk`<br>`nltk.download(` `'stopwords'` `)`<br>`# import nltk for stopwords`<br>`from` `nltk.corpus ` `import` `stopwords`<br>`stop_words ` `=` `set` `(stopwords.words(` `'english'` `))`<br>`# input string `<br>`string ` `=` `"       Python 3.0, released in 2008, was a major revision of the language that is not completely backward compatible and much Python 2 code does not run unmodified on Python 3. With Python 2's end-of-life, only Python 3.6.x[30] and later are supported, with older versions still supporting e.g. Windows 7 (and old installers not restricted to 64-bit Windows)."`<br>`# convert to lower case`<br>`lower_string ` `=` `string.lower()`<br>`# remove numbers`<br>`no_number_string ` `=` `re.sub(r` `'\d+'` `,'',lower_string)`<br>`# remove all punctuation except words and space`<br>`no_punc_string ` `=` `re.sub(r` `'[^\w\s]'` `,'', no_number_string) `<br>`# remove white spaces`<br>`no_wspace_string ` `=` `no_punc_string.strip()`<br>`no_wspace_string`<br>`# convert string to list of words`<br>`lst_string ` `=` `[no_wspace_string][` `0` `].split()`<br>`print` `(lst_string)`<br>`# remove stopwords`<br>`no_stpwords_string` `=` `""`<br>`for` `i ` `in` `lst_string:`<br>`    ``if` `not` `i ` `in` `stop_words:`<br>`        ``no_stpwords_string ` `+` `=` `i` `+` `' '`<br>`        `<br>`# removing last space`<br>`no_stpwords_string ` `=` `no_stpwords_string[:` `-` `1` `]`<br>`# output`<br>`print` `(no_stpwords_string)` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20210113224247/Screenshot346.png)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/how-to-change-default-font-in-tkinter/)

[How to change default font in Tkinter?](https://www.geeksforgeeks.org/how-to-change-default-font-in-tkinter/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

2

Improve

Article Tags :

- [Python](https://www.geeksforgeeks.org/category/programming-language/python/)
- [python-string](https://www.geeksforgeeks.org/tag/python-string/)

Practice Tags :

- [python](https://www.geeksforgeeks.org/explore?category=python)

### Similar Reads

[Text Analysis in Python 3\\
\\
\\
Book's / Document's Content Analysis Patterns within written text are not the same across all authors or languages.This allows linguists to study the language of origin or potential authorship of texts where these characteristics are not directly known such as the Federalist Papers of the American R\\
\\
6 min read](https://www.geeksforgeeks.org/text-analysis-in-python-3/?ref=ml_lbp)
[Exporting PDF Data using Python\\
\\
\\
Sometimes, we have to extract data from PDF. we have to copy & paste the data from PDF. It is time-consuming. In Python, there are packages that we can use to extract data from a PDF and export it in a different format using Python. We will learn how to extract data from PDFs. Extracting Text Wi\\
\\
2 min read](https://www.geeksforgeeks.org/exporting-pdf-data-using-python/?ref=ml_lbp)
[How To Read .Data Files In Python?\\
\\
\\
Unlocking the secrets of reading .data files in Python involves navigating through diverse structures. In this article, we will unravel the mysteries of reading .data files in Python through four distinct approaches. Understanding the structure of .data files is essential, as their format may vary w\\
\\
4 min read](https://www.geeksforgeeks.org/how-to-read-data-files-in-python/?ref=ml_lbp)
[Parsing PDFs in Python with Tika\\
\\
\\
Apache Tika is a library that is used for document type detection and content extraction from various file formats. Using this, one can develop a universal type detector and content extractor to extract both structured text and metadata from different types of documents such as spreadsheets, text do\\
\\
2 min read](https://www.geeksforgeeks.org/parsing-pdfs-in-python-with-tika/?ref=ml_lbp)
[SQL using Python \| Set 3 (Handling large data)\\
\\
\\
It is recommended to go through SQL using Python \| Set 1 and SQL using Python and SQLite \| Set 2 In the previous articles the records of the database were limited to small size and single tuple. This article will explain how to write & fetch large data from the database using module SQLite3 cove\\
\\
4 min read](https://www.geeksforgeeks.org/sql-using-python-set-3-handling-large-data/?ref=ml_lbp)
[How to implement Dictionary with Python3?\\
\\
\\
This program uses python's container called dictionary (in dictionary a key is associated with some information). This program will take a word as input and returns the meaning of that word. Python3 should be installed in your system. If it not installed, install it from this link. Always try to ins\\
\\
3 min read](https://www.geeksforgeeks.org/implement-dictionary-python3/?ref=ml_lbp)
[Printing Lists as Tabular Data in Python\\
\\
\\
During the presentation of data, the question arises as to why the data has to be presented in the form of a table. Tabular data refers to the data that is stored in the form of rows and columns i.e., in the form of a table. It is often preferred to store data in tabular form as data appears more or\\
\\
6 min read](https://www.geeksforgeeks.org/printing-lists-as-tabular-data-in-python/?ref=ml_lbp)
[Replace Commas with New Lines in a Text File Using Python\\
\\
\\
Replacing a comma with a new line in a text file consists of traversing through the file's content and substituting each comma with a newline character. In this article, we will explore three different approaches to replacing a comma with a new line in a text file. Replace Comma With a New Line in a\\
\\
2 min read](https://www.geeksforgeeks.org/replace-commas-with-new-lines-in-a-text-file-using-python/?ref=ml_lbp)
[Last Minute Notes (LMNs) â€“ Data Structures with Python\\
\\
\\
Data Structures and Algorithms (DSA) are fundamental for effective problem-solving and software development. Python, with its simplicity and flexibility, provides a wide range of libraries and packages that make it easier to implement various DSA concepts. This "Last Minute Notes" article offers a q\\
\\
15+ min read](https://www.geeksforgeeks.org/last-minute-notes-lmns-data-structures-with-python/?ref=ml_lbp)
[Visualize data from CSV file in Python\\
\\
\\
CSV stands for Comma-Separated Values, which means that the data in a CSV file is separated by commas, making it easy to store tabular data. The file extension for CSV files is .csv, and these files are commonly used with spreadsheet applications like Google Sheets and Microsoft Excel. A CSV file co\\
\\
4 min read](https://www.geeksforgeeks.org/visualize-data-from-csv-file-in-python/?ref=ml_lbp)

Like2

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/normalizing-textual-data-with-python/)

Improvement

Suggest changes

Suggest Changes

Help us improve. Share your suggestions to enhance the article. Contribute your expertise and make a difference in the GeeksforGeeks portal.

![geeksforgeeks-suggest-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/suggestChangeIcon.png)

Create Improvement

Enhance the article with your expertise. Contribute to the GeeksforGeeks community and help create better learning resources for all.

![geeksforgeeks-improvement-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/createImprovementIcon.png)

Suggest Changes

min 4 words, max Words Limit:1000

## Thank You!

Your suggestions are valuable to us.

## What kind of Experience do you want to share?

[Interview Experiences](https://write.geeksforgeeks.org/posts-new?cid=e8fc46fe-75e7-4a4b-be3c-0c862d655ed0) [Admission Experiences](https://write.geeksforgeeks.org/posts-new?cid=82536bdb-84e6-4661-87c3-e77c3ac04ede) [Career Journeys](https://write.geeksforgeeks.org/posts-new?cid=5219b0b2-7671-40a0-9bda-503e28a61c31) [Work Experiences](https://write.geeksforgeeks.org/posts-new?cid=22ae3354-15b6-4dd4-a5b4-5c7a105b8a8f) [Campus Experiences](https://write.geeksforgeeks.org/posts-new?cid=c5e1ac90-9490-440a-a5fa-6180c87ab8ae) [Competitive Exam Experiences](https://write.geeksforgeeks.org/posts-new?cid=5ebb8fe9-b980-4891-af07-f2d62a9735f2)

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1395486238.1745057513&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=2105257778)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745057513185&cv=11&fst=1745057513185&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fnormalizing-textual-data-with-python%2F&hn=www.googleadservices.com&frm=0&tiba=Normalizing%20Textual%20Data%20with%20Python%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=170259098.1745057513&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

Sign In

By creating this account, you agree to our [Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/) & [Cookie Policy.](https://www.geeksforgeeks.org/legal/privacy-policy/#:~:text=the%20appropriate%20measures.-,COOKIE%20POLICY,-A%20cookie%20is)

# Create Account

Already have an account ?Log in

Continue with Google

or

Username or Email

Password

Institution / Organization

```

```

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)