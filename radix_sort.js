        // Radix Sort Babylon JS webgpu code (based on coumpute shaders)

        const WORKGROUP_SIZE = 64;
        const WORKGROUP_SIZE_2x = WORKGROUP_SIZE * 2;

        function condition(cond, a, b="")
        {
            return cond? a: b;
        }
        
        function get_shader_radix_scan1(has_group_buf, workgroup_size = WORKGROUP_SIZE)
        {
            let workgroup_size_2x = workgroup_size*2;
        
            return  `
                struct Params
                {
                    count: i32,
                    bit: u32
                };
                
                @group(0) @binding(0)
                var<uniform> uParams: Params;
                
                @group(0) @binding(1)
                var<storage, read> bInput : array<i32>;
                
                @group(0) @binding(2)
                var<storage, read_write> bData1 : array<i32>;    
                
                @group(0) @binding(3)
                var<storage, read_write> bData2 : array<i32>;    
                
                ${condition(has_group_buf,`
                @group(0) @binding(4)
                var<storage, read_write> bGroup1 : array<i32>;
                
                @group(0) @binding(5)
                var<storage, read_write> bGroup2 : array<i32>;
                `)}
                
                var<workgroup> s_buf1 : array<i32, ${workgroup_size_2x}>;
                var<workgroup> s_buf2 : array<i32, ${workgroup_size_2x}>;
                
                @compute @workgroup_size(${workgroup_size},1,1)
                fn main(
                    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
                    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
                {
                    let threadIdx = LocalInvocationID.x;
                    let blockIdx = WorkgroupID.x;    
                    let count = arrayLength(&bData1);
                
                    var i = threadIdx + blockIdx*${workgroup_size_2x};
                    if (i<count)
                    {
                        let input = bInput[i];
                        let pred = (input & (1 << uParams.bit)) != 0;
                        s_buf1[threadIdx] = select(1,0,pred);
                        s_buf2[threadIdx] = select(0,1,pred);
                    }
                
                    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
                    if (i<count)
                    {
                        let input = bInput[i];
                        let pred = (input & (1 << uParams.bit)) != 0;
                        s_buf1[threadIdx + ${workgroup_size}] = select(1,0,pred);
                        s_buf2[threadIdx + ${workgroup_size}] = select(0,1,pred);
                    }
                
                    workgroupBarrier();
                
                    var half_size_group = 1u;
                    var size_group = 2u;
                
                    while(half_size_group <= ${workgroup_size})
                    {
                        let gid = threadIdx/half_size_group;
                        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
                        i = tid + blockIdx*${workgroup_size_2x};
                        if (i<count)
                        {
                            s_buf1[tid] = s_buf1[gid*size_group + half_size_group -1] + s_buf1[tid];
                            s_buf2[tid] = s_buf2[gid*size_group + half_size_group -1] + s_buf2[tid];
                        }
                        half_size_group = half_size_group << 1;
                        size_group = size_group << 1;
                        workgroupBarrier();
                    }
                
                    i = threadIdx + blockIdx*${workgroup_size_2x};
                    if (i<count)
                    {
                        bData1[i] = s_buf1[threadIdx];
                        bData2[i] = s_buf2[threadIdx];
                    }
                    
                    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
                    if (i<count)
                    {
                        bData1[i] = s_buf1[threadIdx + ${workgroup_size}];
                        bData2[i] = s_buf2[threadIdx + ${workgroup_size}];
                    }
                
                ${condition(has_group_buf,`        
                    let count_group = arrayLength(&bGroup1);
                    if (threadIdx == 0 && blockIdx<count_group)
                    {        
                        bGroup1[blockIdx] = s_buf1[${workgroup_size_2x} - 1];
                        bGroup2[blockIdx] = s_buf2[${workgroup_size_2x} - 1];
                    }
                `)}
                }
                `;
        }

        class RadixHelper {

        }

        function createBuffer0(size, usage)
        {
            let size_t = (size + 3) & ~3;
            let buffer_out;

            if (!usage) {
                buffer_out = new BABYLON.StorageBuffer(RadixHelper.engine, size_t);
            } else {
                buffer_out = new BABYLON.StorageBuffer(RadixHelper.engine, size_t, usage);
            }

            return buffer_out;
        }

        function GetPipelineRadixScan1_buf(count, has_group_buf, bInput, bData1, bData2, bGroup1, bGroup2) {

            let buf_tmp = new Array(2);

            RadixHelper.RadixScan1_Params_buf = new Array(2);
            //RadixHelper.RadixScan1_CS_buf = new Array(2);

            for(let j=0; j<2; ++j) {
                const cs1 = new BABYLON.ComputeShader("shader_radix_scan1_" + j, RadixHelper.engine, { computeSource: get_shader_radix_scan1(has_group_buf) }, { bindingsMapping:
                    {
                        "uParams": { group: 0, binding: 0 },
                        "bInput": { group: 0, binding: 1 },
                        "bData1": { group: 0, binding: 2 },
                        "bData2": { group: 0, binding: 3 },
                        "bGroup1": { group: 0, binding: 4 },
                        "bGroup2": { group: 0, binding: 5 },
                    }
                });

                cs1.setStorageBuffer("bInput", bInput[j]);
                cs1.setStorageBuffer("bData1", bData1);
                cs1.setStorageBuffer("bData2", bData2);
    
    
                if (has_group_buf) {
                    cs1.setStorageBuffer("bGroup1", bGroup1);
                    cs1.setStorageBuffer("bGroup2", bGroup2);
                } 

                const uBuffer0 = new BABYLON.UniformBuffer(RadixHelper.engine);
          
                uBuffer0.addUniform("count", 1);
                uBuffer0.addUniform("bit", 1);

                cs1.setUniformBuffer("uParams", uBuffer0);
    
                RadixHelper.RadixScan1_Params_buf[j] = uBuffer0;
    
                uBuffer0.updateInt("count", count);
                uBuffer0.update();

                //RadixHelper.RadixScan1_CS_buf[i] = cs1;
    
                buf_tmp[j] = cs1;
            }

            return buf_tmp;
        }

        function Update_PipelineRadixScan1_buf(j, bit_i) {
            RadixHelper.RadixScan1_Params_buf[j].updateUInt("bit", bit_i);
            RadixHelper.RadixScan1_Params_buf[j].update();
        }

        function get_shader_radix_scan2(has_group_buf, workgroup_size = WORKGROUP_SIZE)
        {
            let workgroup_size_2x = workgroup_size*2;
        
            return  `    
                @group(0) @binding(0)
                var<storage, read_write> bData1 : array<i32>;    
                
                @group(0) @binding(1)
                var<storage, read_write> bData2 : array<i32>;    
                
                ${condition(has_group_buf,`
                @group(0) @binding(2)
                var<storage, read_write> bGroup1 : array<i32>;
                
                @group(0) @binding(3)
                var<storage, read_write> bGroup2 : array<i32>;
                `)}
                
                var<workgroup> s_buf1 : array<i32, ${workgroup_size_2x}>;
                var<workgroup> s_buf2 : array<i32, ${workgroup_size_2x}>;
                
                @compute @workgroup_size(${workgroup_size},1,1)
                fn main(
                    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
                    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
                {
                    let threadIdx = LocalInvocationID.x;
                    let blockIdx = WorkgroupID.x;    
                    let count = arrayLength(&bData1);
                
                    var i = threadIdx + blockIdx*${workgroup_size_2x};
                    if (i<count)
                    {
                        s_buf1[threadIdx] = bData1[i];
                        s_buf2[threadIdx] = bData2[i];
                    }
                
                    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
                    if (i<count)
                    {        
                        s_buf1[threadIdx + ${workgroup_size}] = bData1[i];
                        s_buf2[threadIdx + ${workgroup_size}] = bData2[i];
                    }
                
                    workgroupBarrier();
                
                    var half_size_group = 1u;
                    var size_group = 2u;
                
                    while(half_size_group <= ${workgroup_size})
                    {
                        let gid = threadIdx/half_size_group;
                        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
                        i = tid + blockIdx*${workgroup_size_2x};
                        if (i<count)
                        {
                            s_buf1[tid] = s_buf1[gid*size_group + half_size_group -1] + s_buf1[tid];
                            s_buf2[tid] = s_buf2[gid*size_group + half_size_group -1] + s_buf2[tid];
                        }
                        half_size_group = half_size_group << 1;
                        size_group = size_group << 1;
                        workgroupBarrier();
                    }
                
                    i = threadIdx + blockIdx*${workgroup_size_2x};
                    if (i<count)
                    {
                        bData1[i] = s_buf1[threadIdx];
                        bData2[i] = s_buf2[threadIdx];
                    }
                    
                    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
                    if (i<count)
                    {
                        bData1[i] = s_buf1[threadIdx + ${workgroup_size}];
                        bData2[i] = s_buf2[threadIdx + ${workgroup_size}];
                    }
                
                ${condition(has_group_buf,`        
                    let count_group = arrayLength(&bGroup1);
                    if (threadIdx == 0 && blockIdx<count_group)
                    {        
                        bGroup1[blockIdx] = s_buf1[${workgroup_size_2x} - 1];
                        bGroup2[blockIdx] = s_buf2[${workgroup_size_2x} - 1];
                    }
                `)}
                
                }
                `;
        }

        function GetPipelineRadixScan2_buf(buffers_scan1, buffers_scan2) 
        {
            let buf_tmp = new Array(buffers_scan1.length - 1);

            for(let k=1; k<buffers_scan1.length; k++) {

                let has_group_buf = k < buffers_scan1.length - 1;

                const cs2 = new BABYLON.ComputeShader("shader_radix_scan2_" + (k-1), RadixHelper.engine, { computeSource: get_shader_radix_scan2(has_group_buf) }, { bindingsMapping:
                    {
                        "bData1": { group: 0, binding: 0 },
                        "bData2": { group: 0, binding: 1 },
                        "bGroup1": { group: 0, binding: 2 },
                        "bGroup2": { group: 0, binding: 3 },
                        "uParams": { group: 0, binding: 4 },
                    }
                });

                cs2.setStorageBuffer("bData1", buffers_scan1[k]);
                cs2.setStorageBuffer("bData2", buffers_scan2[k]);
    
                if (has_group_buf) {
                    cs2.setStorageBuffer("bGroup1", buffers_scan1[k+1]);
                    cs2.setStorageBuffer("bGroup2", buffers_scan2[k+1]);
                }

                buf_tmp[k-1] = cs2;
            }

            return buf_tmp;
        }

        function get_shader_radix_scan3(workgroup_size = WORKGROUP_SIZE)
        {
            return  ` 
                @group(0) @binding(0)
                var<storage, read_write> bData1 : array<i32>;    

                @group(0) @binding(1)
                var<storage, read_write> bData2 : array<i32>;    

                @group(0) @binding(2)
                var<storage, read> bGroup1 : array<i32>;

                @group(0) @binding(3)
                var<storage, read> bGroup2 : array<i32>;

                @compute @workgroup_size(${workgroup_size},1,1)
                fn main(
                    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
                    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
                {
                    let threadIdx = LocalInvocationID.x;
                    let blockIdx = WorkgroupID.x + 2;    
                    let count = arrayLength(&bData1);

                    let add_idx = WorkgroupID.x / 2;
                    let i = threadIdx + blockIdx*${workgroup_size};

                    {
                        let value = bData1[i];
                        bData1[i] = value + bGroup1[add_idx];
                    }

                    {
                        let value = bData2[i];
                        bData2[i] = value + bGroup2[add_idx];
                    }
                }
                `;
        }

        function GetPipelineRadixScan3_buf(buffers_scan1, buffers_scan2) {

            let buf_tmp = new Array(buffers_scan1.length - 1);

            //let i = 0;
            for (let k = (buffers_scan1.length - 1) - 1; k>=0; k--) {

                const cs3 = new BABYLON.ComputeShader("shader_radix_scan3_" + k, RadixHelper.engine, { computeSource: get_shader_radix_scan3() }, { bindingsMapping:
                    {
                        "bData1": { group: 0, binding: 0 },
                        "bData2": { group: 0, binding: 1 },
                        "bGroup1": { group: 0, binding: 2 },
                        "bGroup2": { group: 0, binding: 3 },
                    }
                });

                cs3.setStorageBuffer("bData1", buffers_scan1[k]);
                cs3.setStorageBuffer("bData2", buffers_scan2[k]);
                cs3.setStorageBuffer("bGroup1", buffers_scan1[k + 1]);
                cs3.setStorageBuffer("bGroup2", buffers_scan2[k + 1]);

                buf_tmp[k] = cs3;

                //buf_tmp[i] = cs3;
                //i++;
            }

            return buf_tmp;
        }

        function get_shader_radix_scatter(workgroup_size = WORKGROUP_SIZE)
        {
            return  `
                @group(0) @binding(0)
                var<uniform> uCount: i32;

                @group(0) @binding(1)
                var<storage, read> bInput : array<i32>;

                @group(0) @binding(2)
                var<storage, read> bIndices1 : array<i32>;

                @group(0) @binding(3)
                var<storage, read> bIndices2 : array<i32>;

                @group(0) @binding(4)
                var<storage, read_write> bOutput : array<i32>;

                @compute @workgroup_size(${workgroup_size},1,1)
                fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
                {
                    // for debug test
                    //bOutput[0] = 16; //uCount; 

                    let idx = i32(GlobalInvocationID.x);
                    if (idx>=uCount)
                    {
                        return;
                    }

                    let value = bInput[idx];
                    if ((idx == 0 && bIndices1[idx]>0) || (idx > 0 && bIndices1[idx]>bIndices1[idx-1]))
                    {
                        bOutput[bIndices1[idx] - 1] = value;
                    }
                    else
                    {
                        let count0 = bIndices1[uCount -1];
                        bOutput[count0 + bIndices2[idx] - 1] = value;
                    }

                    // for debug test
                    //bOutput[0] = 16; //uCount; 
                }   
                `; 
        }

        function GetPipelineRadixScatter_buf(uCount, buf_tmp_in_out, bIndices1, bIndices2)
        {
            let buf_tmp = new Array(2);

            for(let j=0; j<2; ++j) {
                const cs4 = new BABYLON.ComputeShader("shader_radix_scatter" + j, RadixHelper.engine, { computeSource: get_shader_radix_scatter() }, { bindingsMapping:
                    {
                        "uCount": { group: 0, binding: 0 },
                        "bInput": { group: 0, binding: 1 },
                        "bIndices1": { group: 0, binding: 2 },
                        "bIndices2": { group: 0, binding: 3 },
                        "bOutput": { group: 0, binding: 4 },
                    }
                });

                const uBuffer0 = new BABYLON.UniformBuffer(RadixHelper.engine);
                uBuffer0.addUniform("uCount", 1);
                cs4.setUniformBuffer("uCount", uBuffer0);
    
                uBuffer0.updateInt("uCount", uCount); // bool
                uBuffer0.update();
    
                cs4.setStorageBuffer("bInput", buf_tmp_in_out[j]);
                cs4.setStorageBuffer("bIndices1", bIndices1);
                cs4.setStorageBuffer("bIndices2", bIndices2);
                cs4.setStorageBuffer("bOutput", buf_tmp_in_out[1-j]);

                buf_tmp[j] = cs4;
            }

            return buf_tmp;
        }

        function getRandomInt(max) 
        {
            return Math.floor(Math.random() * max);
        }

        function test(engine)
        {
            console.log("RadixSort test(..)::");
            RadixHelper.engine = engine;

            let count = 64*64*64 *4*2; // (~2 MB max) // 64*64*64
            let num_groups_radix_scan = Math.floor((count + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x);
            let max_value = 1073741824-1; // 10000;

            let hInput = new Int32Array(count);
            for (let i=0; i<count; i++)
            {
                hInput[i] = getRandomInt(max_value);
            }

            let hReference = new Int32Array(count);
            hReference.set(hInput);
            hReference.sort();

            let buf_data = createBuffer0(count * 4);

            buf_data.update(hInput, hInput.byteOffset, hInput.byteLength);

            let buf_tmp = new Array(2);
            buf_tmp[0] = createBuffer0(count * 4);
            buf_tmp[1] = createBuffer0(count * 4);

            let buffers_scan1 = [];
            let buffers_scan2 = [];
            let buf_sizes = [];
            let buf_size = count;
            while (buf_size>0)
            {
                let buf1 = createBuffer0(buf_size * 4);
                let buf2 = createBuffer0(buf_size * 4);

                buffers_scan1.push(buf1);
                buffers_scan2.push(buf2);
                buf_sizes.push(buf_size);
                buf_size = Math.floor((buf_size + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x) - 1;
            }

            let bits = 30; // 14;

            let pipeline_radix_scan1_buf = GetPipelineRadixScan1_buf(count, buffers_scan1.length>1, buf_tmp, buffers_scan1[0], buffers_scan2[0], buffers_scan1[1], buffers_scan2[1]);
            let pipeline_radix_scan2_buf = GetPipelineRadixScan2_buf(buffers_scan1, buffers_scan2);
            let pipeline_radix_scan3_buf = GetPipelineRadixScan3_buf(buffers_scan1, buffers_scan2);
            let pipeline_radix_scatter_buf = GetPipelineRadixScatter_buf(count, buf_tmp, buffers_scan1[0],  buffers_scan2[0])

            buf_tmp[0].update(hInput, hInput.byteOffset, hInput.byteLength);

            const startTime = performance.now();

            for (let i=0; i<bits; i++)
            {
                let j = i % 2;
                {
                    {
                        let num_groups = Math.floor((count + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x); 
                        Update_PipelineRadixScan1_buf(j, i);
                        pipeline_radix_scan1_buf[j].dispatch(num_groups, 1, 1);
                    }

                    
                    for (let k = 1; k<buffers_scan1.length; k++)
                    {
                        let num_groups = Math.floor((buf_sizes[k] + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x); 
                        pipeline_radix_scan2_buf[k-1].dispatch(num_groups, 1,1); 
                    }

                    for (let k = (buffers_scan1.length - 1) - 1; k>=0; k--)
                    {
                        let num_groups = Math.floor((buf_sizes[k] + WORKGROUP_SIZE - 1)/WORKGROUP_SIZE) - 2; 
                        pipeline_radix_scan3_buf[k].dispatch(num_groups, 1,1);
                    }
                }

                {
                    let num_groups = Math.floor((count + WORKGROUP_SIZE -1)/WORKGROUP_SIZE);
                    pipeline_radix_scatter_buf[j].dispatch(num_groups, 1,1); 
                }
            }

            const endTime = performance.now();
            console.log(`Call to radix_sort took ${endTime - startTime} milliseconds`);    

            let buf_result;

            {
                let j = bits % 2;
                buf_result = buf_tmp[j];
            }

            let hOutput = new Int32Array(count); 
            {
                buf_result.read(0, count * 4, hOutput).then((arrayBufferView) => {

                    hOutput = arrayBufferView;
                    let count_unmatch = 0;
                    for (let i=0; i<count; i++)
                    {
                        if (hOutput[i] != hReference[i])
                        {
                            count_unmatch++;
                        }
                    }

                    console.log(`count_unmatch: ${count_unmatch}`);
                });
            }

            return endTime - startTime;
        }