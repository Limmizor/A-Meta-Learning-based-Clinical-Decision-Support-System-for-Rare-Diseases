// 添加患者
function addDisease() {
    const form = document.getElementById('addDiseaseForm');
    const formData = new FormData(form);
    
    // 简单的表单验证
    const diseaseName = formData.get('name');
    if (!diseaseName || diseaseName.trim() === '') {
        alert('疾病名称不能为空');
        return;
    }
    
    // 显示加载状态
    const submitBtn = document.querySelector('#addDiseaseModal .btn-primary');
    const originalText = submitBtn.textContent;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> 添加中...';
    submitBtn.disabled = true;
    
    fetch('{{ url_for("disease_management") }}', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            alert('疾病添加成功');
            // 关闭模态框
            const modal = bootstrap.Modal.getInstance(document.getElementById('addDiseaseModal'));
            modal.hide();
            // 刷新页面
            location.reload();
        } else {
            alert('添加失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('错误详情:', error);
        alert('网络错误: ' + error.message);
    })
    .finally(() => {
        // 恢复按钮状态
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    });
}

// 上传医学影像
function uploadImage() {
    const formData = new FormData(document.getElementById('uploadImageForm'));
    
    fetch('/upload_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('影像上传成功');
            $('#uploadImageModal').modal('hide');
            location.reload();
        } else {
            alert('上传失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('发生错误，请重试');
    });
}

// 开始诊断
function startDiagnosis() {
    const formData = new FormData(document.getElementById('diagnoseForm'));
    
    // 显示加载中
    const diagnoseBtn = document.querySelector('#diagnoseModal .btn-primary');
    const originalText = diagnoseBtn.textContent;
    diagnoseBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> 诊断中...';
    diagnoseBtn.disabled = true;
    
    fetch('/diagnose', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('诊断完成');
            $('#diagnoseModal').modal('hide');
            location.reload();
        } else {
            alert('诊断失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('发生错误，请重试');
    })
    .finally(() => {
        // 恢复按钮状态
        diagnoseBtn.textContent = originalText;
        diagnoseBtn.disabled = false;
    });
}

// 图片预览功能
document.addEventListener('DOMContentLoaded', function() {
    // 为所有图片添加点击预览功能
    const images = document.querySelectorAll('.img-thumbnail');
    images.forEach(img => {
        img.addEventListener('click', function() {
            const modal = document.createElement('div');
            modal.className = 'modal fade';
            modal.innerHTML = `
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">影像预览</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body text-center">
                            <img src="${this.src}" class="img-fluid" alt="预览图">
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
            new bootstrap.Modal(modal).show();
            
            // 模态框关闭后移除元素
            modal.addEventListener('hidden.bs.modal', function() {
                document.body.removeChild(modal);
            });
        });
    });
});

// 训练模型
function trainModel() {
    if (confirm('确定要训练模型吗？这可能需要几分钟时间。')) {
        // 显示加载中
        const trainBtn = document.querySelector('.btn-warning');
        const originalText = trainBtn.textContent;
        trainBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> 训练中...';
        trainBtn.disabled = true;
        
        fetch('/train_model', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('模型训练成功');
                location.reload();
            } else {
                alert('模型训练失败: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('发生错误，请重试');
        })
        .finally(() => {
            // 恢复按钮状态
            trainBtn.textContent = originalText;
            trainBtn.disabled = false;
        });
    }
}

// 症状自查提交
function submitSymptomCheck() {
    const formData = new FormData(document.getElementById('symptomForm'));
    
    // 显示加载中
    const submitBtn = document.querySelector('#symptomModal .btn-primary');
    const originalText = submitBtn.textContent;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> 分析中...';
    submitBtn.disabled = true;
    
    fetch('/symptom_check', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('症状自查已提交，系统分析完成');
            $('#symptomModal').modal('hide');
            // 跳转到分析结果页面
            window.location.href = '/symptom_result/' + data.result_id;
        } else {
            alert('提交失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('发生错误，请重试');
    })
    .finally(() => {
        // 恢复按钮状态
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    });
}