../g2d/g2d demo_global.cubin demo_device.cubin _Z9GlobalCpyPfS_i _Z9DeviceCpyPfS_i demo_device_new.cubin
cuobjdump --dump-elf demo_device.cubin > demo_device.dump
cuobjdump --dump-elf demo_device_new.cubin > demo_device_new.dump
diff demo_device.dump demo_device_new.dump
