- [Data Science](https://www.geeksforgeeks.org/data-science-with-python-tutorial/)
- [Data Science Projects](https://www.geeksforgeeks.org/top-data-science-projects/)
- [Data Analysis](https://www.geeksforgeeks.org/data-analysis-tutorial/)
- [Data Visualization](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
- [Machine Learning](https://www.geeksforgeeks.org/machine-learning/)
- [ML Projects](https://www.geeksforgeeks.org/machine-learning-projects/)
- [Deep Learning](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [Computer Vision](https://www.geeksforgeeks.org/computer-vision/)
- [Artificial Intelligence](https://www.geeksforgeeks.org/artificial-intelligence/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/singular-value-decomposition-svd/?type%3Darticle%26id%3D666044&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
True Error vs Sample Error\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/true-error-vs-sample-error/)

# Singular Value Decomposition (SVD)

Last Updated : 04 Feb, 2025

Comments

Improve

Suggest changes

14 Likes

Like

Report

**SVD (Singular Value Decomposition)** is a method used in linear algebra to decompose a matrix into three simpler matrices, making it easier to analyze and manipulate.

## Understanding SVD with Example

Imagine you have a table of data, like a set of ratings where rows are people, and columns are products. The numbers in the table show how much each person likes each product. SVD helps you split that table into three parts:

- **U**: This part tells you about the people (like their general preferences).
- **Σ**: This part shows how important each factor is (how much each rating matters).
- **Vᵀ**: This part tells you about the products (how similar they are to each other)

Suppose you have a small table of people’s ratings for two movies

| Name | Movie 1 Rating | Movie 2 Rating |
| --- | --- | --- |
| Amit | 5 | 3 |
| Sanket | 4 | 2 |
| Harsh | 2 | 5 |

- SVD breaks this table into three smaller parts: one that shows people’s preferences, one that shows the importance of each movie, and one that shows how similar the movies are to each other
- Mathematically, the SVD of a matrix AAA (of size m×nm \\times nm×n) is represented as:

A=UΣVTA = U \\Sigma V^TA=UΣVT

Here:

- UUU: An m×mm \\times mm×m orthogonal matrix whose columns are the left singular vectors of AAA.
- Σ\\SigmaΣ: A diagonal m×nm \\times nm×n matrix containing the singular values of AAA in descending order.
- VTV^TVT: The transpose of an n×nn \\times nn×n orthogonal matrix, where the columns are the right singular vectors of AAA.

## How to perform Singular Value Decomposition

To perform Singular Value Decomposition (SVD) for the matrix A=\[32223−2\]A = \\begin{bmatrix} 3 & 2 & 2 \\\ 2 & 3 & -2 \\end{bmatrix}A=\[32​23​2−2​\], let’s break it down step by step.

**Step 1: Compute** AATA A^TAAT

First, we need to calculate the matrix AATA A^TAAT (where ATA^TAT is the transpose of matrix AAA):

A=\[32223−2\]A = \\begin{bmatrix} 3 & 2 & 2 \\\ 2 & 3 & -2 \\end{bmatrix}A=\[32​23​2−2​\]

AT=\[32232−2\]A^T = \\begin{bmatrix} 3 & 2 \\\ 2 & 3 \\\ 2 & -2 \\end{bmatrix}AT=​322​23−2​​

Now, compute AATA A^TAAT:

AAT=\[32223−2\]⋅\[32232−2\]=\[178817\]A A^T = \\begin{bmatrix} 3 & 2 & 2 \\\ 2 & 3 & -2 \\end{bmatrix} \\cdot \\begin{bmatrix} 3 & 2 \\\ 2 & 3 \\\ 2 & -2 \\end{bmatrix} = \\begin{bmatrix} 17 & 8 \\\ 8 & 17 \\end{bmatrix}AAT=\[32​23​2−2​\]⋅​322​23−2​​=\[178​817​\]

**Step 2: Find the Eigenvalues of** AATA A^TAAT

To find the eigenvalues of AATA A^TAAT, we solve the characteristic equation:

det⁡(AAT–λI)=0\\det(A A^T – \\lambda I) = 0det(AAT–λI)=0

det⁡\[17–λ8817–λ\]=0\\det \\begin{bmatrix} 17 – \\lambda & 8 \\\ 8 & 17 – \\lambda \\end{bmatrix} = 0det\[17–λ8​817–λ​\]=0

(λ–25)(λ–9)=0(\\lambda – 25)(\\lambda – 9) = 0(λ–25)(λ–9)=0

Thus, the eigenvalues are λ1=25\\lambda\_1 = 25λ1​=25 and λ2=9\\lambda\_2 = 9λ2​=9. These eigenvalues correspond to the singular values σ1=5\\sigma\_1 = 5σ1​=5 and σ2=3\\sigma\_2 = 3σ2​=3, since the singular values are the square roots of the eigenvalues.

**Step 3: Find the Right Singular Vectors (Eigenvectors of** ATAA^T AATA **)**

Next, we find the eigenvectors of ATAA^T AATA for λ=25\\lambda = 25λ=25 and λ=9\\lambda = 9λ=9.

#### For λ=25\\lambda = 25λ=25:

Solve (ATA–25I)v=0(A^T A – 25I) v = 0(ATA–25I)v=0:

ATA–25I=\[−1212212−12−22−2−17\]A^T A – 25I = \\begin{bmatrix} -12 & 12 & 2 \\\ 12 & -12 & -2 \\\ 2 & -2 & -17 \\end{bmatrix}ATA–25I=​−12122​12−12−2​2−2−17​​

Row-reduce this matrix to:

\[1−10001000\]\\begin{bmatrix} 1 & -1 & 0 \\\ 0 & 0 & 1 \\\ 0 & 0 & 0 \\end{bmatrix}​100​−100​010​​

The eigenvector corresponding to λ=25\\lambda = 25λ=25 is:

v1=\[12120\]v\_1 = \\begin{bmatrix} \\frac{1}{\\sqrt{2}} \\\ \\frac{1}{\\sqrt{2}} \\\ 0 \\end{bmatrix}v1​=​2​1​2​1​0​​

#### For λ=9\\lambda = 9λ=9:

Solve (ATA–9I)v=0(A^T A – 9I) v = 0(ATA–9I)v=0:

The eigenvector corresponding to λ=9\\lambda = 9λ=9 is:

v2=\[118−118418\]v\_2 = \\begin{bmatrix} \\frac{1}{\\sqrt{18}} \\\ \\frac{-1}{\\sqrt{18}} \\\ \\frac{4}{\\sqrt{18}} \\end{bmatrix}v2​=​18​1​18​−1​18​4​​​

#### For the third eigenvector v3v\_3v3​:

Since v3​ must be perpendicular to v1v\_1v1​ and v2v\_2v2​, we solve the system v1Tv3=0v\_1^T v\_3 = 0v1T​v3​=0 and v2Tv3=0v\_2^T v\_3 = 0v2T​v3​=0, leading to:

v3=\[23−23−13\]v\_3 = \\begin{bmatrix} \\frac{2}{3} \\\ \\frac{-2}{3} \\\ \\frac{-1}{3} \\end{bmatrix}v3​=​32​3−2​3−1​​​

**Step 4: Compute the Left Singular Vectors (Matrix U)**

To compute the left singular vectors U, we use the formula ui=1σiAviu\_i = \\frac{1}{\\sigma\_i} A v\_iui​=σi​1​Avi​. This results in:

U=\[121212−12\]U = \\begin{bmatrix} \\frac{1}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} \\\ \\frac{1}{\\sqrt{2}} & \\frac{-1}{\\sqrt{2}} \\end{bmatrix}U=\[2​1​2​1​​2​1​2​−1​​\]

**Step 5: Final SVD Equation**

Finally, the Singular Value Decomposition of matrix AAA is:

A=UΣVTA = U \\Sigma V^TA=UΣVT

Where:

U=\[121212−12\]U = \\begin{bmatrix} \\frac{1}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} \\\ \\frac{1}{\\sqrt{2}} & \\frac{-1}{\\sqrt{2}} \\end{bmatrix}U=\[2​1​2​1​​2​1​2​−1​​\]

Σ=\[500030\]\\Sigma = \\begin{bmatrix} 5 & 0 & 0 \\\ 0 & 3 & 0 \\end{bmatrix}Σ=\[50​03​00​\]

V=\[12120118−11841823−2313\]V = \\begin{bmatrix} \\frac{1}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} & 0 \\\ \\frac{1}{\\sqrt{18}} & \\frac{-1}{\\sqrt{18}} & \\frac{4}{\\sqrt{18}} \\\ \\frac{2}{3} & \\frac{-2}{3} & \\frac{1}{3} \\end{bmatrix}V=​2​1​18​1​32​​2​1​18​−1​3−2​​018​4​31​​​

Thus, the SVD of matrix AAA is:

A=\[121212−12\]\[500030\]\[12120118−11841823−2313\]A = \\begin{bmatrix} \\frac{1}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} \\\ \\frac{1}{\\sqrt{2}} & \\frac{-1}{\\sqrt{2}} \\end{bmatrix} \\begin{bmatrix} 5 & 0 & 0 \\\ 0 & 3 & 0 \\end{bmatrix} \\begin{bmatrix} \\frac{1}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} & 0 \\\ \\frac{1}{\\sqrt{18}} & \\frac{-1}{\\sqrt{18}} & \\frac{4}{\\sqrt{18}} \\\ \\frac{2}{3} & \\frac{-2}{3} & \\frac{1}{3} \\end{bmatrix}A=\[2​1​2​1​​2​1​2​−1​​\]\[50​03​00​\]​2​1​18​1​32​​2​1​18​−1​3−2​​018​4​31​​​

This is the Result SVD matrix of matrix A.

## Applications of Singular Value Decomposition (SVD)

### 1\. **Calculation of Pseudo-Inverse (Moore-Penrose Inverse)**

The **pseudo-inverse** or **Moore-Penrose inverse** is a generalization of the matrix inverse. It is applicable to matrices that may not be invertible, such as low-rank matrices. When a matrix is invertible, its pseudo-inverse is equal to its inverse. However, the pseudo-inverse exists for matrices that are not necessarily invertible.

The pseudo-inverse of a matrix MMM is denoted as M+M^+M+.

To calculate the pseudo-inverse of a matrix MMM, we first perform the **Singular Value Decomposition (SVD)** of MMM:

M=UΣVTM = U \\Sigma V^TM=UΣVT

Here, UUU and VVV are orthogonal matrices (containing the left and right singular vectors, respectively), and Σ\\SigmaΣ is a diagonal matrix containing the singular values of MMM.

The steps to calculate the pseudo-inverse are as follows:

- **SVD Decomposition**:

M=UΣVTM = U \\Sigma V^TM=UΣVT

- **Multiply both sides by** M−1M^{-1}M−1:

M−1M=M−1UΣVTM^{-1} M = M^{-1} U \\Sigma V^TM−1M=M−1UΣVT

I=M−1UΣVTI = M^{-1} U \\Sigma V^TI=M−1UΣVT

- **Multiply by V**:

VVT=M−1UΣV V^T = M^{-1} U \\SigmaVVT=M−1UΣ

- **Multiply by** W−1W^{-1}W−1 (where Σ\\SigmaΣ is the diagonal matrix of singular values, and the inverse of WWW is W−1=diag(1σ1,1σ2,…,1σn)W^{-1} = \\text{diag}\\left(\\frac{1}{\\sigma\_1}, \\frac{1}{\\sigma\_2}, \\ldots, \\frac{1}{\\sigma\_n}\\right)W−1=diag(σ1​1​,σ2​1​,…,σn​1​)):

VW−1=M−1UVW^{-1} = M^{-1} UVW−1=M−1U

- **Final Equation**:

M+=VW−1UTM^+ = V W^{-1} U^TM+=VW−1UT

Thus, the **pseudo-inverse** of matrix MMM is given by:

M+=VΣ−1UTM^+ = V \\Sigma^{-1} U^TM+=VΣ−1UT

### 2\. **Solving a Set of Homogeneous Linear Equations**

In the case of solving a homogeneous system of linear equations, Mx=bMx = bMx=b:

- If b=0b = 0b=0, we can calculate the SVD and choose any column of VTV^TVT associated with a singular value equal to zero.
- If b≠0b \\neq 0b=0, we can solve Mx=bMx = bMx=b by multiplying both sides by the inverse of MMM:

M−1Mx=M−1bM^{-1} M x = M^{-1}bM−1Mx=M−1b

x=M−1bx = M^{-1} bx=M−1b

Using the pseudo-inverse, we know that:

M−1=VΣ−1UTM^{-1} = V \\Sigma^{-1} U^TM−1=VΣ−1UT

Thus, the solution xxx is:

x=VΣ−1UTbx = V \\Sigma^{-1} U^T bx=VΣ−1UTb

### 3\. **Rank, Range, and Null Space**

The rank, range, and null space of a matrix MMM can be derived from its SVD.

- **Rank**: The rank of matrix MMM is the number of non-zero singular values in Σ\\SigmaΣ.
- **Range**: The range of matrix MMM is the span of the left singular vectors in matrix UUUcorresponding to the non-zero singular values.
- **Null Space**: The null space of matrix MMM is the span of the right singular vectors in matrix VVV corresponding to the zero singular values.

### 4\. **Curve Fitting Problem**

Singular Value Decomposition can be used to **minimize the least square error** in the curve fitting problem. By approximating the solution using the pseudo-inverse, we can find the best-fit curve to a given set of data points.

### 5\. **Applications in Digital Signal Processing (DSP) and Image Processing**

- **Digital Signal Processing**: SVD can be used to analyze signals and filter noise.
- **Image Processing**: SVD is used for image compression and denoising. It helps in reducing the dimensionality of image data by preserving the most significant singular values and discarding the rest.

## Implementation of Singular Value Decomposition (SVD)

In this code, we will try to calculate the Singular value decomposition using Numpy and Scipy.  We will be calculating SVD, and also performing pseudo-inverse. In the end, we can apply SVD for compressing the image

Python`
from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
X = np.array([[3, 3, 2], [2, 3, -2]])
print(X)
U, singular, V_transpose = svd(X)
print("U: ", U)
print("Singular array", singular)
print("V^{T}", V_transpose)
singular_inv = 1.0 / singular
s_inv = np.zeros(X.shape)
s_inv[0][0] = singular_inv[0]
s_inv[1][1] = singular_inv[1]
M = np.dot(np.dot(V_transpose.T, s_inv.T), U.T)
print(M)
# SVD on cat Image
cat = data.chelsea()
plt.imshow(cat)
gray_cat = rgb2gray(cat)
U, S, V_T = svd(gray_cat, full_matrices=False)
S = np.diag(S)
fig, ax = plt.subplots(5, 2, figsize=(8, 20))
curr_fig = 0
for r in [5, 10, 70, 100, 200]:
    cat_approx = U[:, :r] @ S[0:r, :r] @ V_T[:r, :]
    ax[curr_fig][0].imshow(cat_approx, cmap='gray')
    ax[curr_fig][0].set_title("k = "+str(r))
    ax[curr_fig, 0].axis('off')
    ax[curr_fig][1].set_title("Original Image")
    ax[curr_fig][1].imshow(gray_cat, cmap='gray')
    ax[curr_fig, 1].axis('off')
    curr_fig += 1
plt.show()
`

**Output:**

```
[[ 3  3  2]\
 [ 2  3 -2]]
---------------------------
U:  [[-0.7815437 -0.6238505]\
 [-0.6238505  0.7815437]]
---------------------------
Singular array [5.54801894 2.86696457]
---------------------------
V^{T} [[-0.64749817 -0.7599438  -0.05684667]\
 [-0.10759258  0.16501062 -0.9804057 ]\
 [-0.75443354  0.62869461  0.18860838]]
--------------------------
# Inverse
array([[ 0.11462451,  0.04347826],\
       [ 0.07114625,  0.13043478],\
       [ 0.22134387, -0.26086957]])
---------------------------
```

![](https://media.geeksforgeeks.org/wp-content/uploads/20230403085120/svd_compressed.png)

Original image vs SVD k-image

The output consists of subplots showing the compressed image for different values of **r (5, 10, 70, 100, 200)**, where r represents the number of singular values used in the approximation. As the value of r increases, the compressed image becomes closer to the original grayscale image of the cat, with smaller values of r leading to more blurred and blocky images, and larger values retaining more details.

## Conclusion

In conclusion, the SVD decomposition function simplifies complex data by breaking it into three smaller parts. This helps uncover hidden patterns and relationships, making it easier to analyze and work with large datasets. SVD is useful in tasks like recommendations, data compression, and finding important features, making data simpler and more manageable.

[iframe](https://cdnads.geeksforgeeks.org/instream/video.html)

Singular Value Decomposition (SVD) in Machine Learning

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/true-error-vs-sample-error/)

[True Error vs Sample Error](https://www.geeksforgeeks.org/true-error-vs-sample-error/)

[P](https://www.geeksforgeeks.org/user/pawangfg/)

[pawangfg](https://www.geeksforgeeks.org/user/pawangfg/)

Follow

14

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [DSA](https://www.geeksforgeeks.org/category/dsa/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Mathematical](https://www.geeksforgeeks.org/category/dsa/algorithm/mathematical/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [python](https://www.geeksforgeeks.org/tag/python/)

+2 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [Mathematical](https://www.geeksforgeeks.org/explore?category=Mathematical)
- [python](https://www.geeksforgeeks.org/explore?category=python)

### Similar Reads

[Singular Value Decomposition\\
\\
\\
Prerequisites: Matrix Diagonalization, Eigenvector Computation and Low-Rank Approximations Before getting in depth into the SVD, let us first briefly understand what Matrix Diagonalization technique is and when it fails to perform efficiently. Matrix Diagonalization Matrix diagonalization is the pro\\
\\
8 min read](https://www.geeksforgeeks.org/singular-value-decomposition/)
[LU Decomposition\\
\\
\\
LU decomposition or factorization of a matrix is the factorization of a given square matrix into two triangular matrices, one upper triangular matrix and one lower triangular matrix, such that the product of these two matrices gives the original matrix. It was introduced by Alan Turing in 1948, who\\
\\
8 min read](https://www.geeksforgeeks.org/l-u-decomposition-system-linear-equations/)
[Partial Least Squares Singular Value Decomposition (PLSSVD)\\
\\
\\
Partial Least Squares Singular Value Decomposition (PLSSVD) is a sophisticated statistical technique employed in the realms of multivariate analysis and machine learning. This method merges the strengths of Partial Least Squares (PLS) and Singular Value Decomposition (SVD), offering a powerful tool\\
\\
9 min read](https://www.geeksforgeeks.org/partial-least-squares-singular-value-decomposition-plssvd/)
[QR Decomposition in Machine learning\\
\\
\\
QR decomposition is a way of expressing a matrix as the product of two matrices: Q (an orthogonal matrix) and R (an upper triangular matrix). In this article, I will explain decomposition in Linear Algebra, particularly QR decomposition among many decompositions. What is QR Decomposition?Decompositi\\
\\
9 min read](https://www.geeksforgeeks.org/qr-decomposition-in-machine-learning/)
[Faces dataset decompositions in Scikit Learn\\
\\
\\
The Faces dataset is a database of labeled pictures of people's faces that can be found in the well-known machine learning toolkit Scikit-Learn. Face recognition, facial expression analysis, and other computer vision applications are among the frequent uses for it. The Labeled Faces in the Wild (LFW\\
\\
5 min read](https://www.geeksforgeeks.org/faces-dataset-decompositions-in-scikit-learn/)
[Choleski Decomposition in R\\
\\
\\
Cholesky Decomposition is a popular numerical method used in linear algebra for decomposing a Hermitian positive-definite matrix into the product of a lower triangular matrix and its transpose. In this article, we'll learn how to perform Cholesky Decomposition in R programming language. Before divin\\
\\
4 min read](https://www.geeksforgeeks.org/choleski-decomposition-in-r/)
[Doolittle Algorithm \| LU Decomposition\\
\\
\\
Doolittle Algorithm: The Doolittle Algorithm is a method for performing LU Decomposition, where a given matrix is decomposed into a lower triangular matrix L and an upper triangular matrix U. This decomposition is widely used in solving systems of linear equations, inverting matrices, and computing\\
\\
11 min read](https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/)
[Understanding Cross Decomposition in Machine Learning\\
\\
\\
Usually, in real-world datasets, some of the features of the data are highly correlated with each other. Applying normal regression methods to highly correlated data is not an effective way to analyze such data, since multicollinearity makes the estimates highly sensitive to any change in the model.\\
\\
15 min read](https://www.geeksforgeeks.org/understanding-cross-decomposition-in-machine-learning/)
[Cholesky Decomposition : Matrix Decomposition\\
\\
\\
In linear algebra, a matrix decomposition or matrix factorization is a factorization of a matrix into a product of matrices. There are many different matrix decompositions. One of them is Cholesky Decomposition. What is Cholesky Decomposition?The Cholesky decomposition or Cholesky factorization is a\\
\\
8 min read](https://www.geeksforgeeks.org/cholesky-decomposition-matrix-decomposition/)
[Gaussian Elimination to Solve Linear Equations\\
\\
\\
The Gaussian Elimination Method is a widely used technique for solving systems of linear equations. A system of linear equations involves multiple equations with unknown variables. The goal of solving such a system is to find the values of these unknowns that satisfy all the given equations simultan\\
\\
15+ min read](https://www.geeksforgeeks.org/gaussian-elimination/)

Like14

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/singular-value-decomposition-svd/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=2064036003.1745056701&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=2038284173)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745056701669&cv=11&fst=1745056701669&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fsingular-value-decomposition-svd%2F&hn=www.googleadservices.com&frm=0&tiba=Singular%20Value%20Decomposition%20(SVD)%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1585534731.1745056702&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=normal&cb=nf085q3cohz2)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=normal&cb=37s79mw6013s)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=invisible&cb=bzxqec55rlre)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)