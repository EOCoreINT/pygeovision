{{- define "pygeovision.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "pygeovision.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (include "pygeovision.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "pygeovision.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{ include "pygeovision.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{- define "pygeovision.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pygeovision.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
