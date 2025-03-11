package controllers

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/Azure/azure-storage-blob-go/azblob"
	"github.com/go-logr/logr"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/yaml"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	mlopsv1alpha1 "github.com/yourusername/azure-blob-k8s-operator/api/v1alpha1"
)

// AzureBlobWatcherReconciler reconciles a AzureBlobWatcher object
type AzureBlobWatcherReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=mlops.sertel.com,resources=azureblobwatchers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=mlops.sertel.com,resources=azureblobwatchers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=mlops.sertel.com,resources=azureblobwatchers/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=argoproj.io,resources=workflows,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch

func (r *AzureBlobWatcherReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("azureblobwatcher", req.NamespacedName)

	// Fetch the AzureBlobWatcher instance
	instance := &mlopsv1alpha1.AzureBlobWatcher{}
	err := r.Get(ctx, req.NamespacedName, instance)
	if err != nil {
		if errors.IsNotFound(err) {
			// Request object not found, could have been deleted
			return ctrl.Result{}, nil
		}
		// Error reading the object - requeue the request
		return ctrl.Result{}, err
	}

	// Get credentials from Secret
	secret := &corev1.Secret{}
	err = r.Get(ctx, types.NamespacedName{
		Name:      instance.Spec.CredentialsSecret,
		Namespace: req.Namespace,
	}, secret)
	if err != nil {
		log.Error(err, "Failed to get Azure Storage credentials secret")
		// Update status with error
		r.updateStatusWithError(ctx, instance, err)
		return ctrl.Result{RequeueAfter: time.Minute}, nil
	}

	// Set up Azure Blob Storage connection
	var credential azblob.Credential
	var storageURL string

	if accountKey, ok := secret.Data["accountKey"]; ok {
		credential, err = azblob.NewSharedKeyCredential(
			instance.Spec.StorageAccount,
			string(accountKey),
		)
		if err != nil {
			log.Error(err, "Failed to create shared key credential")
			r.updateStatusWithError(ctx, instance, err)
			return ctrl.Result{RequeueAfter: time.Minute}, nil
		}
		storageURL = fmt.Sprintf("https://%s.blob.core.windows.net", instance.Spec.StorageAccount)
	} else if sasToken, ok := secret.Data["sasToken"]; ok {
		credential = azblob.NewAnonymousCredential()
		storageURL = fmt.Sprintf("https://%s.blob.core.windows.net/%s%s",
			instance.Spec.StorageAccount,
			instance.Spec.ContainerName,
			string(sasToken))
	} else {
		err = fmt.Errorf("neither accountKey nor sasToken found in secret %s", instance.Spec.CredentialsSecret)
		log.Error(err, "Invalid secret content")
		r.updateStatusWithError(ctx, instance, err)
		return ctrl.Result{RequeueAfter: time.Minute}, nil
	}

	// Create pipeline and container client
	pipeline := azblob.NewPipeline(credential, azblob.PipelineOptions{})
	serviceURL := azblob.NewServiceURL(storageURL, pipeline)
	containerURL := serviceURL.NewContainerURL(instance.Spec.ContainerName)

	// List blobs
	listBlobs, err := containerURL.ListBlobsHierarchySegment(ctx, azblob.Marker{}, "/", azblob.ListBlobsSegmentOptions{
		Prefix: instance.Spec.BlobPrefix,
	})
	if err != nil {
		log.Error(err, "Failed to list blobs")
		r.updateStatusWithError(ctx, instance, err)
		return ctrl.Result{RequeueAfter: time.Minute}, nil
	}

	// Process new blobs
	var newBlobs []string
	var processedBlobs []string

	// Compile blob pattern if provided
	var blobPattern *regexp.Regexp
	if instance.Spec.BlobPattern != "" {
		blobPattern, err = regexp.Compile(instance.Spec.BlobPattern)
		if err != nil {
			log.Error(err, "Invalid blob pattern regex", "pattern", instance.Spec.BlobPattern)
			r.updateStatusWithError(ctx, instance, err)
			return ctrl.Result{RequeueAfter: time.Minute}, nil
		}
	}

	// Track last detected blobs
	lastDetectedBlobs := make(map[string]bool)
	for _, blobName := range instance.Status.LastDetectedBlobs {
		lastDetectedBlobs[blobName] = true
	}

	// Check for new blobs
	for _, blob := range listBlobs.Segment.BlobItems {
		blobName := blob.Name
		// Apply regex filter if configured
		if blobPattern != nil && !blobPattern.MatchString(blobName) {
			continue
		}

		// Check if we've already seen this blob
		if !lastDetectedBlobs[blobName] {
			newBlobs = append(newBlobs, blobName)
		}
		processedBlobs = append(processedBlobs, blobName)
	}

	// Process new blobs
	for _, blobName := range newBlobs {
		log.Info("New blob detected", "blobName", blobName)
		
		// Generate blob URL (may need SAS token or other auth)
		blobURL := fmt.Sprintf("https://%s.blob.core.windows.net/%s/%s",
			instance.Spec.StorageAccount,
			instance.Spec.ContainerName,
			blobName)
		
		// Process template
		err = r.processTemplateAndCreateResource(ctx, instance, blobName, blobURL, req.Namespace)
		if err != nil {
			log.Error(err, "Failed to process template for blob", "blobName", blobName)
			continue
		}
	}

	// Update status
	if len(newBlobs) > 0 || len(instance.Status.LastDetectedBlobs) != len(processedBlobs) {
		instance.Status.LastChecked = metav1.Now()
		instance.Status.LastDetectedBlobs = processedBlobs
		
		if err := r.Status().Update(ctx, instance); err != nil {
			log.Error(err, "Failed to update AzureBlobWatcher status")
			return ctrl.Result{}, err
		}
	}

	// Schedule next reconciliation based on polling interval
	return ctrl.Result{
		RequeueAfter: time.Duration(instance.Spec.PollingIntervalSeconds) * time.Second,
	}, nil
}

// Process template and create corresponding K8s resource
func (r *AzureBlobWatcherReconciler) processTemplateAndCreateResource(
	ctx context.Context,
	instance *mlopsv1alpha1.AzureBlobWatcher,
	blobName string,
	blobURL string,
	namespace string,
) error {
	// Replace template variables
	template := instance.Spec.JobTemplate.Template
	template = strings.ReplaceAll(template, "${BLOB_NAME}", blobName)
	template = strings.ReplaceAll(template, "${BLOB_URL}", blobURL)
	
	// Process based on resource kind
	switch instance.Spec.JobTemplate.Kind {
	case "Job":
		return r.createJob(ctx, template, instance, blobName, namespace)
	case "Workflow":
		return r.createWorkflow(ctx, template, instance, blobName, namespace)
	default:
		return fmt.Errorf("unsupported resource kind: %s", instance.Spec.JobTemplate.Kind)
	}
}

// Create a Kubernetes Job
func (r *AzureBlobWatcherReconciler) createJob(
	ctx context.Context,
	template string,
	instance *mlopsv1alpha1.AzureBlobWatcher,
	blobName string,
	namespace string,
) error {
	job := &batchv1.Job{}
	if err := yaml.Unmarshal([]byte(template), job); err != nil {
		return fmt.Errorf("failed to unmarshal Job template: %w", err)
	}
	
	// Make sure the job has the right namespace
	job.Namespace = namespace
	
	// Set owner reference
	if err := ctrl.SetControllerReference(instance, job, r.Scheme); err != nil {
		return fmt.Errorf("failed to set controller reference: %w", err)
	}
	
	// Create the Job
	if err := r.Create(ctx, job); err != nil {
		if errors.IsAlreadyExists(err) {
			// Job already exists, not an error
			return nil
		}
		return fmt.Errorf("failed to create Job: %w", err)
	}
	
	// Update status with triggered job
	r.addTriggeredJob(ctx, instance, blobName, job.Name)
	return nil
}

// Create an Argo Workflow
func (r *AzureBlobWatcherReconciler) createWorkflow(
	ctx context.Context,
	template string,
	instance *mlopsv1alpha1.AzureBlobWatcher,
	blobName string,
	namespace string,
) error {
	// Parse the template into a generic map
	var workflow map[string]interface{}
	if err := yaml.Unmarshal([]byte(template), &workflow); err != nil {
		return fmt.Errorf("failed to unmarshal Workflow template: %w", err)
	}
	
	// Set namespace
	workflow["metadata"].(map[string]interface{})["namespace"] = namespace
	
	// Convert to unstructured
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		return fmt.Errorf("failed to marshal workflow: %w", err)
	}
	
	// Create dynamic client for Argo Workflow CRD
	dynamicClient, err := client.New(ctrl.GetConfigOrDie(), client.Options{})
	if err != nil {
		return fmt.Errorf("failed to create dynamic client: %w", err)
	}
	
	// Create unstructured object
	unstructured := &client.UnstructuredContent{}
	if err := json.Unmarshal(workflowJSON, unstructured); err != nil {
		return fmt.Errorf("failed to unmarshal to unstructured: %w", err)
	}
	
	// Create the Workflow
	if err := dynamicClient.Create(ctx, unstructured); err != nil {
		if errors.IsAlreadyExists(err) {
			// Workflow already exists, not an error
			return nil
		}
		return fmt.Errorf("failed to create Workflow: %w", err)
	}
	
	// Update status with triggered workflow
	workflowName, _, _ := unstructured.NestedString("metadata", "name")
	r.addTriggeredJob(ctx, instance, blobName, workflowName)
	return nil
}

// Add a triggered job to the status
func (r *AzureBlobWatcherReconciler) addTriggeredJob(
	ctx context.Context,
	instance *mlopsv1alpha1.AzureBlobWatcher,
	blobName string,
	jobName string,
) {
	// Add job to status
	triggeredJob := mlopsv1alpha1.TriggeredJob{
		BlobName:     blobName,
		JobName:      jobName,
		CreationTime: metav1.Now(),
		Status:       "Created",
	}
	
	instance.Status.TriggeredJobs = append(instance.Status.TriggeredJobs, triggeredJob)
	
	// Update status
	if err := r.Status().Update(ctx, instance); err != nil {
		r.Log.Error(err, "Failed to update status with triggered job")
	}
}

// Update status with error
func (r *AzureBlobWatcherReconciler) updateStatusWithError(
	ctx context.Context,
	instance *mlopsv1alpha1.AzureBlobWatcher,
	err error,
) {
	condition := metav1.Condition{
		Type:               "Error",
		Status:             metav1.ConditionTrue,
		Reason:             "ReconciliationError",
		Message:            err.Error(),
		LastTransitionTime: metav1.Now(),
	}
	
	// Update conditions
	instance.Status.Conditions = append(instance.Status.Conditions, condition)
	instance.Status.LastChecked = metav1.Now()
	
	if err := r.Status().Update(ctx, instance); err != nil {
		r.Log.Error(err, "Failed to update status with error condition")
	}
}

// SetupWithManager sets up the controller with the Manager
func (r *AzureBlobWatcherReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&mlopsv1alpha1.AzureBlobWatcher{}).
		Complete(r)
}
