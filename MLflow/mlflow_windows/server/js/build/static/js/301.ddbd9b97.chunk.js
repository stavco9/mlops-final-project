(self.webpackChunk_mlflow_mlflow=self.webpackChunk_mlflow_mlflow||[]).push([[301],{33150:function(t,e,a){"use strict";a.r(e),a.d(e,{default:function(){return u}});var r=a(91604),n=a(66433),s=a(471),o=a(28014),i=a(86987),d=a(26530),c=a(73564),l=a(65069);n.v0.GlobalWorkerOptions.workerSrc="./static-files/pdf.worker.js";class h extends r.Component{constructor(){super(...arguments),this.state={loading:!0,error:void 0,pdfData:void 0,currentPage:1,numPages:1},this.onDocumentLoadSuccess=t=>{let{numPages:e}=t;this.setState({numPages:e})},this.onDocumentLoadError=t=>{d.default.logErrorAndNotifyUser(new c.V(t))},this.onPageChange=(t,e)=>{this.setState({currentPage:t})},this.renderPdf=()=>(0,l.tZ)(r.Fragment,{children:(0,l.BX)("div",{className:"pdf-viewer",children:[(0,l.tZ)("div",{className:"paginator",children:(0,l.tZ)(s.tlE,{simple:!0,currentPageIndex:this.state.currentPage,numTotal:this.state.numPages,pageSize:1,onChange:this.onPageChange,dangerouslySetAntdProps:{simple:!0}})}),(0,l.tZ)("div",{className:"document",children:(0,l.tZ)(n.BB,{file:this.state.pdfData,onLoadSuccess:this.onDocumentLoadSuccess,onLoadError:this.onDocumentLoadError,loading:(0,l.tZ)(o.S,{}),children:(0,l.tZ)(n.T3,{pageNumber:this.state.currentPage,loading:(0,l.tZ)(o.S,{})})})})]})})}fetchPdf(){const t=(0,i.Oz)(this.props.path,this.props.runUuid);this.props.getArtifact(t).then((t=>{this.setState({pdfData:{data:t},loading:!1})})).catch((t=>{this.setState({error:t,loading:!1})}))}componentDidMount(){this.fetchPdf()}componentDidUpdate(t){this.props.path===t.path&&this.props.runUuid===t.runUuid||this.fetchPdf()}render(){return this.state.loading?(0,l.tZ)("div",{className:"artifact-pdf-view-loading",children:"Loading..."}):this.state.error?(0,l.tZ)("div",{className:"artifact-pdf-view-error",children:"Oops we couldn't load your file because of an error. Please reload the page to try again."}):(0,l.tZ)("div",{className:"pdf-outer-container",children:this.renderPdf()})}}h.defaultProps={getArtifact:i.e0};var u=h},70172:function(){},2001:function(){},33779:function(){},82258:function(){}}]);
//# sourceMappingURL=301.ddbd9b97.chunk.js.map