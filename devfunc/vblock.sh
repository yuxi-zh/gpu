
function VirtualBlock {
	sass=${1};
	new_sass=${2};
	in_schedule=False;
	while IFS='' read -r line || [[ -n "${line}" ]]; do
		if [[ ${line} =~ .*\<\\CONSTANT_MAPPING\>.* ]]; then
			echo "vblock : c[0x0][0x0]" >> ${new_sass};
			echo ${line} >> ${new_sass};
		elif [[ ${line} =~ .*\<SCHEDULE_BLOCK\>.* ]]; then
			in_schedule=True;
			echo ${line} >> ${new_sass};
		elif [[ ${line} =~ .*\<\\SCHEDULE_BLOCK\>.* ]]; then
			in_schedule=False;
			echo ${line} >> ${new_sass};
		elif [[ ${line} =~ .*S2R.*SR_CTAID ]]; then
			read SRN SR <<< `echo ${line} | sed -E 's/.*S2R (.*),\s+(SR_CTAID\..);/\1 \2/'`
			echo "SR=${SR}, SRN=${SRN}"
			if [[ ${in_schedule} == False ]]; then
				echo '<SCHEDULE_BLOCK>' >> ${new_sass};
			fi
			echo "--:-:-:-:- S2R ${SRN}, ${SR};" >> ${new_sass};
			echo "--:-:-:-:- ISCADD ${SRN}, ${SRN}, vblock, 0x2;" >> ${new_sass};
			echo "--:-:-:-:- LDG ${SRN}, [${SRN}];" >> ${new_sass};
			if [[ ${in_schedule} == False ]]; then
				echo '<\SCHEDULE_BLOCK>' >> ${new_sass};
			fi
		else
			echo ${line} >> ${new_sass};
		fi
	done < ${sass}
}

function TestVirtualBlock {
	case0="
	--:-:-:-:0      MOV one, 1;
	--:-:1:-:6      S2R tid, SR_TID.X;
	--:-:-:Y:d      ISETP.EQ.AND P0, PT, one, param_RST, PT;
	--:-:-:-:5  @P0 BRA.U CTAID1;
	--:-:2:-:1      S2R blkMPQ, SR_CTAID.X;
	--:-:3:-:1      S2R blkI,   SR_CTAID.Y;
	--:-:4:-:1      S2R blkE,   SR_CTAID.Z;
	--:-:-:-:5      BRA.U END_CTAID1;
	"
	case1='
	<SCHEDULE_BLOCK>
	--:-:-:-:0      MOV one, 1;
	--:-:1:-:6      S2R tid, SR_TID.X;
	--:-:-:Y:d      ISETP.EQ.AND P0, PT, one, param_RST, PT;
	--:-:-:-:5  @P0 BRA.U CTAID1;
	--:-:2:-:1      S2R blkMPQ, SR_CTAID.X;
	--:-:3:-:1      S2R blkI,   SR_CTAID.Y;
	--:-:4:-:1      S2R blkE,   SR_CTAID.Z;
	--:-:-:-:5      BRA.U END_CTAID1;
	<\SCHEDULE_BLOCK>
	'
	case2='
	<CONSTANT_MAPPING>
	<\CONSTANT_MAPPING>
	'
	printf "${case0}" | tee tmp0.txt;
	printf "" | tee tmp1.txt;
	VirtualBlock tmp0.txt tmp1.txt;
	cat tmp0.txt tmp1.txt;
	rm tmp0.txt tmp1.txt;
	printf "${case1}" | tee tmp0.txt;
	printf "" | tee tmp1.txt;
	VirtualBlock tmp0.txt tmp1.txt;
	cat tmp0.txt tmp1.txt;
	printf "${case2}" | tee tmp0.txt;
	printf "" | tee tmp1.txt;
	VirtualBlock tmp0.txt tmp1.txt;
	cat tmp0.txt tmp1.txt;
}

function GenerateVirtualBlockSass {
	mkdir -p kernels/vblock;
	for sass in `find kernels/sass -name *.sass`; do
		new_sass=`echo ${sass} | sed 's/sass/vblock/'`
		echo ${new_sass};
		VirtualBlock ${sass} ${new_sass}
	done
}

# TestVirtualBlock
GenerateVirtualBlockSass