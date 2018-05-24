
cd kernels/sass;

for sass in `ls`; do
	sed -i 's/EXIT/RET/g' ${sass};
done

cd ../..;