# Solution for Nonogram with CUDA C

## 0. Introduce

This is my project when I take the course of Parallel Programming. I wanted to solve puzzle which name is a Nonogram using CUDA C.  There's a lot of code for solving this puzzle but it is hard to find the version of CUDA. That's what I wanted to start the project with this subject.

### 0.1 What is the Nonogram
Nonogram is the puzzle game which is developed in Japan and it has other names such as griddlers, picross, and logic puzzle. Generally, there is a given grid with arbitrary size and several numbers that are located on the left side and upper side of the grid. The numbers indicate the hint of each row and column, which implies the number of cells to be colored. The aim of this puzzle is that making pictures according to the hints of each row and column. (See Fig. 1.)  Here is the rule for the puzzle.  
- __Rule 1__: Color the cells consecutively according to the hints.  
- __Rule 2__: At least make one blank between the hints.
- __Rule 3__: The order of the hints is identical with the order of the colored cells.
![fig1](https://user-images.githubusercontent.com/55457315/149344461-f0e92b48-990b-4c79-923a-b02981b3e244.png)



__0.1.1 Input__  
The first line of the input gives two integers, let N and M. N is the size of rows and M is the size of columns. The next N lines of input which consist of an arbitrary number of integers inform how many cells will be colored from the 1st row to Nth row. The next M lines of input inform how many cells will be colored from 1st column to Mth column.

__0.1.2 Output__  
The output will show the hidden picture. For convinience, I would use white and black box, which correspond to blank and hint. 
![image](https://user-images.githubusercontent.com/55457315/149344626-cd8d91a4-4ddc-4c63-86f6-13390066878d.png)


### 0.2 How to Solve it?  
There is a lot of effort to solve the Nonogram with computer programming with lots of techniques, and most of them seek for efficient method only using CPU. For example, one can conduct solving nonograms with the backtracking method which determines the possibility of the next row based on the current row case. Because the backtracking method does not scan every case for the puzzle, it can reduce searching time. However, this method works serial way, it is hard to apply in a parallel way. For solving the nonogram with the parallel method, I use straightforward but possible to do a parallel way, which is using the basic technique for solving a puzzle. Therefore, I tried to use straight-forward but easily parallable strategy.  It is basic technique for solving puzzle. Let’s see the steps.  
- __Step 1__: Find every possible case for each row and column.  
- __Step 2__: Compare every case and find common cell.  
- __Step 3__: Reduce case according to the common cell from Step 2.  
- __Step 4__: Repeat Step 2 and 3 until puzzle is solved.   

For example, one can find every possible case for the row of width 5 and with hints 1 and 2. (See Fig. 3. a.) Since there are only three possible cases and one can find only 4th cell is colored in every case so it can be determined that the 4th cell of this row will be colored. After that, using a determined cell from row hints, it is possible to reduce the possibility and determine whether some cells will be colored or not. Assume the hint of the 4th column is 3. (See Fig. 3. b.) It is easy to figure out that there are 2 possible cases, one is that to color from the first cell to the third cell and the other is that to color from the second cell to the last cell. Since the first cell of the column is colored by the previous step, the only possible way is to color this column from the first cell. The only thing we have to solve this puzzle is that repeat these methods until the puzzle is solved. It has a limitation that considers every possible case for rows and columns so it would take more time than the backtracking method when conducting serial ways. However, it can make a parallel way for the most part of the method and because consider every case, I expect that it will make an accurate result with reducing much time.
![image](https://user-images.githubusercontent.com/55457315/149344752-68c780d3-0471-4716-b23f-7f7798075e1a.png)

## 1 Implementation
### 1.1 Initializing
**1. Reading input file in the host and allocate memory to device **
![image](https://user-images.githubusercontent.com/55457315/149345272-f5a176bb-0051-48fa-ac97-9a4e303acc57.png)

**2. Make data structure for Nonogram data **
![image](https://user-images.githubusercontent.com/55457315/149345435-d3d64348-24dd-44c0-a2c6-a5184841e594.png)

**3. Find every case for each row and columns **
![image](https://user-images.githubusercontent.com/55457315/149345729-cbf200bf-d51d-4f5b-b51b-c41ef32c6f65.png)  
![image](https://user-images.githubusercontent.com/55457315/149345749-564df912-b86d-43f6-8a10-3ca11f9cf450.png)  
![image](https://user-images.githubusercontent.com/55457315/149345763-36e3d060-7978-4b6d-afe2-29bffc4c1471.png)  
![image](https://user-images.githubusercontent.com/55457315/149345947-d784f16c-0ef6-49e8-be5a-ccf3a182cd21.png)

### 1.2 Solve
![image](https://user-images.githubusercontent.com/55457315/149346061-0002f6b2-7906-4c1c-8092-ae60d6da8c6d.png)

### 1.3 Result
![image](https://user-images.githubusercontent.com/55457315/149346156-1791a99f-1036-4b9b-b7dd-339d8f013656.png)  
![image](https://user-images.githubusercontent.com/55457315/149346193-1b6c8836-c7ef-4bd6-9cf9-63407dae1c3b.png)  
![image](https://user-images.githubusercontent.com/55457315/149346331-b6979729-c390-4069-8d99-12a0989ec1bb.png)  
![image](https://user-images.githubusercontent.com/55457315/149346383-05c97cdb-52c5-4a93-be47-5f9097a8f993.png)



