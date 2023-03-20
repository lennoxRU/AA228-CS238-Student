
statess = collect(1:10)
probss = zeros(10)
probss[1]=0.5
probss[2]=0.5
dist = SparseCat(statess, probss)
print(dist)

