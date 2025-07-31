//let annotations = new Array(totalProblems).fill(null).map(() => Object.fromEntries(all_metrics.map(metric => [metric, null])));
// annotations is a dictionary of dictionaries, with the outer dictionary indexed by uid and the inner dictionary indexed by metric
let annotations = {}
all_uids.forEach(key => {
    annotations[key] = {};
    all_metrics.forEach(metric => {
        annotations[key][metric] = null;
    });
});

function getAnnotationCount() {
    let count = 0;
    for (let i = 0; i < all_uids.length; i++) {
        if (annotations[all_uids[i]]['example'] !== null && annotations[all_uids[i]]['code'] !== null) {
            count += 1;
        }
    }
    return count;
}
function loadAnnotations() {
    const savedAnnotations = localStorage.getItem('annotations');
    if (savedAnnotations) {
        prev_annotations = JSON.parse(savedAnnotations);
        // get all notations from the previous annotations with the same uid
        for (let i = 0; i < all_uids.length; i++) {
            const uid = all_uids[i];
            // check if the uid is in the previous annotations
            if (uid in prev_annotations) {
                annotations[uid] = prev_annotations[uid];
            }
        }
        annotatedCount = getAnnotationCount();
        updateProgress();
        updateButtons();
    }
}

function saveAnnotations() {
    localStorage.setItem('annotations', JSON.stringify(annotations));
}

function showProblem(index) {
    document.querySelector(`#problem_${currentProblem}`).style.display = 'none';
    document.querySelector(`#problem_${index}`).style.display = 'block';
    currentProblem = index;
    updateProgress();
    updateButtons();
}

function nextProblem() {
    if (currentProblem < totalProblems - 1) {
        showProblem(currentProblem + 1);
    }
}

function prevProblem() {
    if (currentProblem > 0) {
        showProblem(currentProblem - 1);
    }
}

function annotate(metric_name, annotation, index) {
    annotations[all_uids[index]][metric_name] = annotation;
    annotatedCount = getAnnotationCount();
    saveAnnotations();
    updateProgress();
    updateButtons();
}

function updateProgress() {
    document.getElementById('progress').textContent = `${annotatedCount}/${totalProblems}`;
}

function updateButtons() {
    const exampleGoodButton = document.getElementById(`example_good_${currentProblem}`);
    const exampleOkButton = document.getElementById(`example_ok_${currentProblem}`);
    const exampleBadButton = document.getElementById(`example_bad_${currentProblem}`);
    const codeGoodButton = document.getElementById(`code_good_${currentProblem}`);
    const codeOkButton = document.getElementById(`code_ok_${currentProblem}`);
    const codeBadButton = document.getElementById(`code_bad_${currentProblem}`);
    exampleGoodButton.classList.remove('pressed');
    exampleOkButton.classList.remove('pressed');
    exampleBadButton.classList.remove('pressed');
    codeGoodButton.classList.remove('pressed');
    codeOkButton.classList.remove('pressed');
    codeBadButton.classList.remove('pressed');
    if (annotations[all_uids[currentProblem]]['example'] === 'good') {
        exampleGoodButton.classList.add('pressed');
    } else if (annotations[all_uids[currentProblem]]['example'] === 'ok') {
        exampleOkButton.classList.add('pressed');
    } else if (annotations[all_uids[currentProblem]]['example'] === 'bad') {
        exampleBadButton.classList.add('pressed');
    }
    if (annotations[all_uids[currentProblem]]['code'] === 'good') {
        codeGoodButton.classList.add('pressed');
    } else if (annotations[all_uids[currentProblem]]['code'] === 'ok') {
        codeOkButton.classList.add('pressed');
    } else if (annotations[all_uids[currentProblem]]['code'] === 'bad') {
        codeBadButton.classList.add('pressed');
    }
}

function download_to_file(filename, codeBase64) {
    const code = atob(codeBase64);
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}`;
    a.click();
    URL.revokeObjectURL(url);
}

function download_div_to_image(div_id, filename) {
    const div = document.getElementById(div_id);
    const loading = document.createElement('div');
    loading.textContent = 'Generating image...';
    loading.style.position = 'fixed';
    loading.style.fontSize = '32px';
    loading.style.fontWeight = 'bold';
    loading.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
    loading.style.padding = '10px';
    loading.style.borderRadius = '5px';
    loading.style.left = '50%';
    loading.style.top = '50%';
    loading.style.transform = 'translate(-50%, -50%)';
    loading.style.backdropFilter = 'blur(5px)';
    document.body.appendChild(loading);

    setTimeout(() => {
        html2canvas(div).then(canvas => {
            document.body.removeChild(loading);
            const image = canvas.toDataURL('image/png');
            const a = document.createElement('a');
            a.href = image;
            a.download = filename;
            a.click();
        }).catch(error => {
            const loadingElement = document.getElementById('loadingPopup');
            if (loadingElement) {
                document.body.removeChild(loadingElement);
            }
            console.error('Error generating image:', error);
        });
    }, 100);
}

function download_with_annotations(filename, jsonBase64, uid) {
    const jsonData = JSON.parse(atob(jsonBase64));
    newJsonData = {
        "uid": jsonData.uid,
        "annotations": annotations[uid],
        "metadata": jsonData.metadata,
        "examples": jsonData.examples,
        "code": jsonData.code,
    };
    const updatedJsonBase64 = btoa(JSON.stringify(newJsonData, null, 2));
    download_to_file(filename, updatedJsonBase64);
}

window.onload = function() {
    loadAnnotations();
    showProblem(0);
};